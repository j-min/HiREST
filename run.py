import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from packaging import version
import numpy as np
from tqdm import tqdm
import torch
import logging
import json
from trainer_base import TrainerBase
from args import get_parser

from hirest_dataset import get_moment_loader, MultitaskLoader
from hirest_dataset import frame_index_to_timestamp

from utils import LossMeter, set_global_logging_level
import dist_utils

set_global_logging_level(logging.ERROR, ["transformers"])

_use_native_amp = True
from torch.cuda.amp import autocast

class Trainer(TrainerBase):
    def __init__(self, args, train=True):
        super().__init__(
            args,
            )

        self.wandb_initialized = False
        
        from modeling import MomentModel

        tasks = []

        if args.task_moment_retrieval:
            tasks.append('moment_retrieval')
        if args.task_moment_segmentation:
            tasks.append('moment_segmentation')
        if args.task_step_captioning:
            tasks.append('step_captioning')
        self.tasks = tasks
        print('tasks:', self.tasks)

        asr_dim = -1
        if 'clip' in args.asr_feature_dir.lower():
            asr_dim = 512
        elif 'minilm'in args.asr_feature_dir.lower():
            asr_dim = 384


        model = MomentModel(
            n_frames=args.n_model_frames,
            asr_dim=asr_dim,
            args=args,
        )
        self.model = model

        if self.verbose:
            # total paramters in M
            total_params = sum(p.numel() for p in self.model.parameters())
            # number of trainable paramters in M
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f'# total parameters: {total_params / 1e6:.2f}M')
            print(f'# trainable parameters: {trainable_params / 1e6:.2f}M')

        self.build_loaders()

        # Load Checkpoint
        self.start_epoch = None
        if args.load is not None:
            if not str(args.load).endswith('.pth'):
                ckpt_path = args.load + '.pth'
            else:
                ckpt_path = args.load
            self.load_checkpoint(ckpt_path)

        # GPU Options
        print(f'Model Launching at GPU {self.args.gpu}')
        if self.verbose:
            from time import time
            start = time()
        self.model = self.model.to(args.gpu)

        # Optimizer
        if train:
            self.optim, self.lr_scheduler = self.create_optimizer_and_scheduler()

            if self.args.fp16 and _use_native_amp:
                self.scaler = torch.cuda.amp.GradScaler()

        if args.distributed:
            self.model = DDP(self.model, device_ids=[args.gpu],
                                find_unused_parameters=True
                                )
        if self.verbose:
            print(f'It took {time() - start:.1f}s')

    def build_loaders(self):

        args = self.args

        gpu = args.local_rank

        print(f'Building train loader at GPU {gpu}')

        if args.train:
            loaders = []
            for task in self.tasks:
                loader = get_moment_loader(args, split='train', batch_size=args.train_batch_size, task=task)
                loaders.append(loader)
            train_loader = MultitaskLoader(loaders, verbose=self.verbose, shuffle=True)
            self.train_loader = train_loader
            if self.verbose:
                print('# len(train_loader):', len(train_loader))


            print(f'Building val loader at GPU {gpu}')

            if 'moment_retrieval' in self.tasks:
                self.val_moment_retrieval_loader = get_moment_loader(
                    args,
                    split='val',
                    batch_size=args.eval_batch_size,
                    task='moment_retrieval',
                )

            if 'moment_segmentation' in self.tasks:

                self.val_moment_segmentation_loader = get_moment_loader(
                    args,
                    split='val',
                    batch_size=args.eval_batch_size,
                    task='moment_segmentation',
                )

            if 'step_captioning' in self.tasks:

                self.val_step_captioning_loader = get_moment_loader(
                    args,
                    split='val',
                    batch_size=args.eval_batch_size,
                    task='step_captioning',
                )


        print(f'Building test loader at GPU {gpu}')

        if args.end_to_end:
            if 'moment_retrieval' in self.tasks:
                self.test_moment_retrieval_loader = get_moment_loader(
                    args,
                    split='test',
                    batch_size=args.eval_batch_size,
                    task='moment_retrieval',
                )
            elif 'moment_segmentation' in self.tasks:
                self.test_moment_segmentation_loader = get_moment_loader(
                    args,
                    split='test',
                    batch_size=args.eval_batch_size,
                    task='moment_segmentation',
                )
            elif 'step_captioning' in self.tasks:

                self.test_step_captioning_loader = get_moment_loader(
                    args,
                    split='test',
                    batch_size=args.eval_batch_size,
                    task='step_captioning',
                )
        else:
            if 'moment_retrieval' in self.tasks:
                self.test_moment_retrieval_loader = get_moment_loader(
                    args,
                    split='test',
                    batch_size=args.eval_batch_size,
                    task='moment_retrieval',
                )

            if 'moment_segmentation' in self.tasks:
                self.test_moment_segmentation_loader = get_moment_loader(
                    args,
                    split='test',
                    batch_size=args.eval_batch_size,
                    task='moment_segmentation',
                )

            if 'step_captioning' in self.tasks:

                self.test_step_captioning_loader = get_moment_loader(
                    args,
                    split='test',
                    batch_size=args.eval_batch_size,
                    task='step_captioning',
                )

    def train(self):
        if args.train:
            if self.verbose:
                loss_meter = LossMeter()
                best_valid = 0.
                best_epoch = 0

                if not self.wandb_initialized:

                    self.wandb_initialized = True

            if self.args.distributed:
                dist.barrier()

            global_step = 0
            epochs = self.args.epochs

            for epoch in range(epochs):

                if self.start_epoch is not None:
                    epoch += self.start_epoch
                self.train_loader.set_epoch(epoch)

                self.model.train()
                if self.args.distributed:
                    self.model.module.freeze_clip()
                else:
                    self.model.freeze_clip()
                if self.verbose:
                    pbar = tqdm(total=len(self.train_loader), ncols=120)

                epoch_results = {
                    'loss': 0.,

                }

                task_counter = {}
                for task in self.tasks:
                    task_counter[task] = 0

                for step_i, batch in enumerate(self.train_loader):

                    task = batch['tasks'][0]
                    task_counter[task] += 1

                    if self.args.fp16 and _use_native_amp:
                        with autocast():
                            if self.args.distributed:
                                results = self.model.module.train_step(batch)
                            else:
                                results = self.model.train_step(batch)
                    else:
                        if self.args.distributed:
                            results = self.model.module.train_step(batch)
                        else:
                            results = self.model.train_step(batch)

                    loss = results['loss']

                    if self.args.fp16 and _use_native_amp:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    loss = loss.detach()

                    # Update Parameters
                    if self.args.clip_grad_norm > 0:
                        if self.args.fp16 and _use_native_amp:
                            self.scaler.unscale_(self.optim)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.args.clip_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self.args.clip_grad_norm)

                    update = True
                    if self.args.gradient_accumulation_steps > 1:
                        if step_i == 0:
                            update = False
                        elif step_i % self.args.gradient_accumulation_steps == 0 or step_i == len(self.train_loader) - 1:
                            update = True
                        else:
                            update = False

                    if update:
                        if self.args.fp16 and _use_native_amp:
                            self.scaler.step(self.optim)
                            self.scaler.update()
                        else:
                            self.optim.step()

                        if self.lr_scheduler:
                            self.lr_scheduler.step()

                        for param in self.model.parameters():
                            param.grad = None
                        global_step += 1

                    for k, v in results.items():
                        if k in epoch_results:
                            epoch_results[k] += v.item()

                    if self.lr_scheduler:
                        if version.parse(torch.__version__) >= version.parse("1.4"):
                            lr = self.lr_scheduler.get_last_lr()[0]
                        else:
                            lr = self.lr_scheduler.get_lr()[0]
                    else:
                        try:
                            lr = self.optim.get_lr()[0]
                        except AttributeError:
                            lr = self.args.lr

                    if self.verbose:
                        loss_meter.update(loss.item())
                        desc_str = f'Epoch {epoch} | LR {lr:.6f} | Steps {global_step}'
                        desc_str += f' | Loss {loss_meter.val:4f}'
                        pbar.set_description(desc_str)
                        pbar.update(1)

                if self.args.distributed:
                    dist.barrier()

                if self.verbose:
                    pbar.close()

                val_loss = 0

                if 'moment_retrieval' in self.tasks:
                    val_moment_retrieval_results = self.evaluate(self.val_moment_retrieval_loader, has_target=True)
                    val_loss += val_moment_retrieval_results['loss']

                if 'moment_segmentation' in self.tasks:

                    val_moment_segmentation_results = self.evaluate(self.val_moment_segmentation_loader, has_target=False)

                if 'step_captioning' in self.tasks:
                    val_step_captioning_results = self.evaluate(self.val_step_captioning_loader, has_target=True)
                    val_loss += val_step_captioning_results['loss']

                if self.verbose:
                    print(f'Epoch {epoch} | Val Loss {val_loss:.4f}')
                    if 'moment_retrieval' in self.tasks:
                        print(f'Moment Retrieval loss: {val_moment_retrieval_results["loss"]:.4f}')

                    if 'step_captioning' in self.tasks:
                        print(f'Step Captioning loss: {val_step_captioning_results["loss"]:.4f}')

                    if val_loss < best_valid or epoch == 0:
                        best_valid = val_loss
                        best_epoch = epoch
                        self.save("BEST")

                    if 'moment_retrieval' in self.tasks:
                        moment_retrieval_dump_path = os.path.join(self.args.ckpt_dir, f'moment_retrieval_epoch_{str(epoch).zfill(3)}.json')
                        with open(moment_retrieval_dump_path, 'w') as f:
                            json.dump(val_moment_retrieval_results, f, indent=4)
                            print(f'Saved {moment_retrieval_dump_path}')

                    if 'moment_segmentation' in self.tasks:
                        moment_segmentation_dump_path = os.path.join(self.args.ckpt_dir, f'moment_segmentation_epoch_{str(epoch).zfill(3)}.json')
                        with open(moment_segmentation_dump_path, 'w') as f:
                            json.dump(val_moment_segmentation_results, f, indent=4)
                            print(f'Saved {moment_segmentation_dump_path}')

                    if 'step_captioning' in self.tasks:
                        step_captioning_dump_path = os.path.join(self.args.ckpt_dir, f'step_captioning_epoch_{str(epoch).zfill(3)}.json')
                        with open(step_captioning_dump_path, 'w') as f:
                            json.dump(val_step_captioning_results, f, indent=4)
                            print(f'Saved {step_captioning_dump_path}')

                if self.args.distributed:
                    dist.barrier()

            if self.verbose:
                self.save("LAST")

            epoch = best_epoch
            print('Best Epoch: ', epoch)

        best_path = os.path.join(self.args.ckpt_dir, 'BEST')

        self.load(best_path, loc="cpu")

        if args.end_to_end:
            import shutil

            shutil.copyfile(f"{args.data_dir}/all_data_test.json", f"{args.data_dir}/all_data_test_original.json")

            if 'moment_retrieval' in self.tasks:
                moments = self.evaluate(self.test_moment_retrieval_loader, has_target=False)

                moment_retrieval_dump_path = os.path.join(self.args.ckpt_dir, f'test_moment_retrieval_end_to_end.json')
                with open(moment_retrieval_dump_path, 'w') as f:
                    json.dump(moments, f, indent=4)
                    print(f'Saved {moment_retrieval_dump_path}')
                
                shutil.copyfile(f"{args.data_dir}/all_data_test.json", f"{args.data_dir}/temp1.json")

                with open(f"{args.data_dir}/temp1.json", 'r') as f:
                    test = json.load(f)

                    for prompt in test:
                        if not (prompt in moments):
                            continue
                        
                        for video in test[prompt]:
                            if not (video in moments[prompt]):
                                continue
                            
                            test[prompt][video]["bounds"] = moments[prompt][video]["bounds"]

                            test[prompt][video]["steps"] = [ ]

                            for i in range(5):
                                test[prompt][video]["steps"].append(
                                    {"index": i, "heading": "", "absolute_bounds": [i, i+1]}
                                )
                
                with open(f"{args.data_dir}/all_data_test.json", 'w') as f:
                    json.dump(test, f, indent=2)

                if 'moment_segmentation' in self.tasks:
                    self.test_moment_segmentation_loader = get_moment_loader(
                        args,
                        split='test',
                        batch_size=args.eval_batch_size,
                        task='moment_segmentation',
                    )

            if 'moment_segmentation' in self.tasks:
                moments = self.evaluate(self.test_moment_segmentation_loader, has_target=False)

                moment_segmentation_dump_path = os.path.join(self.args.ckpt_dir, f'test_moment_segmentation_end_to_end.json')
                with open(moment_segmentation_dump_path, 'w') as f:
                    json.dump(moments, f, indent=4)

                shutil.copyfile(f"{args.data_dir}/all_data_test.json", f"{args.data_dir}/temp2.json")

                with open(f"{args.data_dir}/temp2.json", 'r') as f:
                    test = json.load(f)

                    for prompt in test:
                        for video in test[prompt]:
                            test[prompt][video]["steps"] = []
                            
                            if not (video in moments):
                                continue

                            for i, bound in enumerate(moments[video]["bounds"]):
                                test[prompt][video]["steps"].append({
                                    "index": i,
                                    "heading": "",
                                    "absolute_bounds": bound
                                })

                with open(f"{args.data_dir}/all_data_test.json", 'w') as f:
                    json.dump(test, f, indent=2)

                if 'step_captioning' in self.tasks:
                    self.test_step_captioning_loader = get_moment_loader(
                        args,
                        split='test',
                        batch_size=args.eval_batch_size,
                        task='step_captioning',
                    )
            
            if 'step_captioning' in self.tasks:
                moments = self.evaluate(self.test_step_captioning_loader, has_target=False)

                step_captioning_dump_path = os.path.join(self.args.ckpt_dir, f'test_step_captioning_end_to_end.json')
                with open(step_captioning_dump_path, 'w') as f:
                    json.dump(moments, f, indent=4)

                with open(f"{args.data_dir}/all_data_test.json", 'r') as f:
                    test = json.load(f)

                    for prompt in test:
                        for video in test[prompt]:
                            if video in moments:
                                for i, sent in enumerate(moments[video]["captions"]):
                                    test[prompt][video]["steps"][i]["heading"] = sent["sentence"]


                with open(f"{self.args.ckpt_dir}/final_end_to_end_results.json", 'w') as f:
                    json.dump(test, f, indent=2)
                    print(f"Final results saved too: {self.args.ckpt_dir}/final_end_to_end_results.json !")

                
                shutil.move(f"{args.data_dir}/all_data_test.json", f"{args.data_dir}/temp3.json")
                
            shutil.move(f"{args.data_dir}/all_data_test_original.json", f"{args.data_dir}/all_data_test.json")

        else:
            if 'moment_retrieval' in self.tasks:
                test_moment_retrieval_results = self.evaluate(self.test_moment_retrieval_loader, has_target=False)
            if 'moment_segmentation' in self.tasks:
                test_moment_segmentation_results = self.evaluate(self.test_moment_segmentation_loader, has_target=False)
            if 'step_captioning' in self.tasks:
                test_step_captioning_results = self.evaluate(self.test_step_captioning_loader, has_target=False)

            if self.verbose:
                if 'moment_retrieval' in self.tasks:
                    moment_retrieval_dump_path = os.path.join(self.args.ckpt_dir, f'test_moment_retrieval_BEST.json')
                    with open(moment_retrieval_dump_path, 'w') as f:
                        json.dump(test_moment_retrieval_results, f, indent=4)
                        print(f'Saved {moment_retrieval_dump_path}')
                if 'moment_segmentation' in self.tasks:
                    moment_segmentation_dump_path = os.path.join(self.args.ckpt_dir, f'test_moment_segmentation_BEST.json')
                    with open(moment_segmentation_dump_path, 'w') as f:
                        json.dump(test_moment_segmentation_results, f, indent=4)
                        print(f'Saved {moment_segmentation_dump_path}')
                if 'step_captioning' in self.tasks:
                    step_captioning_dump_path = os.path.join(self.args.ckpt_dir, f'test_step_captioning_BEST.json')
                    with open(step_captioning_dump_path, 'w') as f:
                        json.dump(test_step_captioning_results, f, indent=4)
                        print(f'Saved {step_captioning_dump_path}')


        if self.args.distributed:
            dist.barrier()

    def predict(self, loader, has_target=False):
        """Predict the results.
        """
        self.model.eval()
        with torch.no_grad():

            predictions = []
            targets = []

            gen_kwargs = {}
            gen_kwargs['num_beams'] = self.args.num_beams

            losses = []
            moments = []
            video_fnames = []
            tasks = []
            prompts = []
            approximate_bounds = []
            original_bounds = []
            video_duration = []

            boundary_scores = []

            task = loader.task

            for i, batch in enumerate(tqdm(loader, ncols=120, desc=f"{task.capitalize()} Prediction", disable=not self.verbose)):

                if has_target:
                    if self.args.fp16 and _use_native_amp:
                        with autocast():
                            if self.args.distributed:
                                results = self.model.module.train_step(batch)
                            else:
                                results = self.model.train_step(batch)
                    else:
                        if self.args.distributed:
                            results = self.model.module.train_step(batch)
                        else:
                            results = self.model.train_step(batch)

                    loss = results['loss']
                    loss = loss.detach().item()

                if self.args.distributed:
                    results = self.model.module.test_step(
                        batch,
                        **gen_kwargs)
                else:
                    results = self.model.test_step(
                        batch,
                        **gen_kwargs)

                predictions.extend(results['prediction'])

                if task == 'moment_retrieval':
                    start_end_target = torch.cat([
                        batch['moment_retrieval_start_target'].view(-1, 1),
                        batch['moment_retrieval_end_target'].view(-1, 1),
                    ], dim=1).cpu().detach().tolist()
                    targets.extend(start_end_target)

                elif task == 'moment_segmentation':
                    boundary_target = batch['all_bound_frames']
                    targets.extend(boundary_target)

                elif task == 'step_captioning':
                    if "target_text_raw" in batch:
                        targets.extend(batch["target_text_raw"])
                    else:
                        targets.extend(batch['target_text'])

                if 'video_fnames' in batch:
                    video_fnames.extend(batch['video_fnames'])

                tasks.extend(batch['tasks'])
                prompts.extend(batch['prompts'])

                if 'video_duration' in batch:
                    video_duration.extend(batch['video_duration'])

                if 'boundary_scores' in results:
                    boundary_scores.extend(results['boundary_scores'])

                if has_target:
                    losses.append(loss)

            results = {
                'tasks': tasks,
                'prompts': prompts,
                'predictions': predictions,
            }

            if has_target:
                results['targets'] = targets
                results['loss'] = losses
            elif task == 'moment_segmentation':
                results['targets'] = targets

            if len(boundary_scores) > 0:
                results['boundary_scores'] = boundary_scores

            if len(video_fnames) > 0:
                results['video_fnames'] = video_fnames

            if len(approximate_bounds) > 0:
                results['approximate_bounds'] = approximate_bounds

            if len(original_bounds) > 0:
                results['original_bounds'] = original_bounds

            if len(video_duration) > 0:
                results['video_duration'] = video_duration


            if self.args.distributed:
                dist.barrier()

                dist_results = dist_utils.all_gather(results)

                results = {}

                predictions = []
                targets = []
                tasks = []
                video_fnames = []
                prompts = []
                video_duration = []
                approximate_bounds = []
                original_bounds = []

                boundary_scores = []

                for result in dist_results:
                    predictions.extend(result['predictions'])
                    tasks.extend(result['tasks'])
                    video_fnames.extend(result['video_fnames'])
                    prompts.extend(result['prompts'])

                    video_duration.extend(result['video_duration'])

                    if 'boundary_scores' in result:
                        boundary_scores.extend(result['boundary_scores'])

                results['predictions'] = predictions
                results['tasks'] = tasks
                results['video_fnames'] = video_fnames
                results['prompts'] = prompts

                results['video_duration'] = video_duration

                results['boundary_scores'] = boundary_scores

                if has_target:
                    losses = []
                    for result in dist_results:
                        losses.extend(result['loss'])
                        targets.extend(result['targets'])
                    results['targets'] = targets
                elif task == 'moment_segmentation':
                    for result in dist_results:
                        targets.extend(result['targets'])
                    results['targets'] = targets

                if self.verbose:
                    print('after all_gather - len(predictions)', len(results['predictions']))

                dist.barrier()

            assert len(results['tasks']) == len(results['video_fnames']) == len(results['prompts']) \
                , f"len(tasks)={len(results['tasks'])}, len(video_fnames)={len(results['video_fnames'])}, len(prompts)={len(results['prompts'])}"

            if has_target:
                assert len(results['predictions']) == len(results['video_fnames'])

            results['loss'] = np.mean(losses)

            if has_target:
                loss = results['loss']
                targets = results['targets']
            elif task == 'moment_segmentation':
                targets = results['targets']


            if tasks[0] == 'moment_retrieval':
                moment_retrieval_results = {}
                # Moment retrieval
                # {
                #     "prompt1": {
                #         "video1": {
                #             "bounds": [ 10, 210 ],
                #             "...": "..."
                #         },
                #         "video2": {
                #             "bounds": [ 60, 70 ],
                #             "...": "..."
                #         }
                #     }
                # }
                for i in range(len(results['video_fnames'])):

                    prompt = results['prompts'][i]
                    video_fname = results['video_fnames'][i]
                    if prompt not in moment_retrieval_results:
                        moment_retrieval_results[prompt] = {}

                    if video_fname not in moment_retrieval_results[prompt]:
                        moment_retrieval_results[prompt][video_fname] = {}

                    raw_prediction = results['predictions'][i]
                    assert len(raw_prediction) == 2
                    start = frame_index_to_timestamp(raw_prediction[0], results['video_duration'][i], n_frames=self.args.n_model_frames)
                    end = frame_index_to_timestamp(raw_prediction[1], results['video_duration'][i], n_frames=self.args.n_model_frames)

                    moment_retrieval_results[prompt][video_fname]['bounds'] = [start, end]

                    video_duration = results['video_duration'][i]
                    moment_retrieval_results[prompt][video_fname]['video_duration'] = video_duration

                    if has_target:
                        start_end_target = results['targets'][i]
                        moment_retrieval_results[prompt][video_fname]['target_bounds'] = start_end_target

                if has_target:
                    moment_retrieval_results['loss'] = loss

                return moment_retrieval_results

            elif tasks[0] == 'moment_segmentation':
                moment_segmentation_results = {}
                # moment_segmentation
                # {
                #     "video": {
                #         "captions": [ { "sentence": "I like small dogs", "...": "..." }, { ".
                # ..": "..." } ],
                #         "bounds": [ [1, 10.4], [20, 26], [ "..." ] ]
                #     }
                # }
                for i in range(len(results['tasks'])):
                    video_fname = results['video_fnames'][i]
                    if video_fname not in moment_segmentation_results:
                        moment_segmentation_results[video_fname] = {}

                    raw_prediction = results['predictions'][i]

                    bounds = []
                    for j in range(len(raw_prediction) - 1):
                        bound = []
                        try:
                            bound.append(frame_index_to_timestamp(raw_prediction[j], results['video_duration'][i], n_frames=self.args.n_model_frames))
                            bound.append(frame_index_to_timestamp(raw_prediction[j+1], results['video_duration'][i], n_frames=self.args.n_model_frames))
                        except:
                            print(f"Video: {results['video_fnames'][i]} | Bound {raw_prediction[j]} or {raw_prediction[j+1]} out of {results['video_duration'][i]}")

                        bounds.append(bound)

                    moment_segmentation_results[video_fname]['bounds'] = bounds

                    video_duration = results['video_duration'][i]
                    moment_segmentation_results[video_fname]['video_duration'] = video_duration

                    moment_segmentation_results[video_fname]['pred_bounds'] = raw_prediction
                    moment_segmentation_results[video_fname]['target_bounds'] = results['targets'][i]

                if has_target:
                    moment_segmentation_results['loss'] = loss

                return moment_segmentation_results




            elif tasks[0] == 'step_captioning':
                step_captioning_results = {}
                # Step captioning
                # {
                #     "video": {
                #         "captions": [ { "sentence": "I like small dogs", "...": "..." }, { ".
                # ..": "..." } ],
                #         "bounds": [ [1, 10.4], [20, 26], [ "..." ] ]
                #     }
                # }

                for i in range(len(results['tasks'])):

                    video_fname = results['video_fnames'][i]
                    if video_fname not in step_captioning_results:
                        step_captioning_results[video_fname] = {}

                    if 'captions' not in step_captioning_results[video_fname]:
                        step_captioning_results[video_fname]['captions'] = []


                    raw_prediction = results['predictions'][i]

                    step_captioning_results[video_fname]['captions'].append({
                        'sentence': raw_prediction,
                    })


                    video_duration = results['video_duration'][i]
                    step_captioning_results[video_fname]['video_duration'] = video_duration

                    if has_target:
                        target_text = results['targets'][i]
                        
                        if 'target_captions' not in step_captioning_results[video_fname]:
                            step_captioning_results[video_fname]['target_captions'] = []

                        step_captioning_results[video_fname]['target_captions'].append(target_text)


                if has_target:
                    step_captioning_results['loss'] = loss

                return step_captioning_results

            else:
                raise ValueError('Unknown task: {}'.format(tasks[0]))


    def evaluate(self, loader, dump_path=None, has_target=False):
        results = self.predict(loader, has_target=has_target)
        return results


def main_worker(gpu, args):
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {gpu}')

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl')

    trainer = Trainer(args, train=args.train)
    trainer.train()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    from accelerate.utils import set_seed
    import random
    import numpy as np

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = 0
    args.local_rank = local_rank
    args.gpu = local_rank

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    if local_rank in [0, -1]:
        print(args)

        comments = []
        if args.load is not None:
            ckpt_str = "_".join(args.load.split('/')[-3:])
            comments.append(ckpt_str)
        if args.comment != '':
            comments.append(args.comment)
        comment = '_'.join(comments)

        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M')

        run_name = f'{current_time}_GPU{args.world_size}'
        if len(comments) > 0:
            run_name += f'_{comment}'

        args.run_name = run_name

    if args.distributed:
        main_worker(local_rank, args)
    else:
        main_worker(0, args)


