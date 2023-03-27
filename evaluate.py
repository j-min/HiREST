import json
import language_evaluation
import numpy as np
from tqdm import tqdm
import argparse

def load_data(gt_data, pred_data):
    gt = gt_data
    pred = pred_data

    if isinstance(gt_data, str):
        with open(gt_data, 'r') as f:
            gt = json.load(f)
    else:
        assert isinstance(gt, dict), "GT data should be a str path or a dict"
    
    if isinstance(pred_data, str):
        with open(pred_data, 'r') as f:
            pred = json.load(f)
    else:
        assert isinstance(pred, dict), "Prediction data should be a str path or a dict"

    return gt, pred

def compute_iou(interval_1, interval_2):
    start_i, end_i = interval_1[0], interval_1[1]
    start, end = interval_2[0], interval_2[1]
    intersection = max(0, min(end, end_i) - max(start, start_i))
    union = min(max(end, end_i) - min(start, start_i), end-start + end_i-start_i)
    iou = float(intersection) / (union + 1e-8)
    return iou

def evaluate_video_retrieval(gt_data, pred_data):
    gt, pred = load_data(gt_data, pred_data)

    ks = [ 1, 5, 10, 50 ]
    count = { }
    total = { }
    
    for cat in PROMPT_CATEGORIES:
        count[cat] = {}
        total[cat] = 0

        for k in ks:
            count[cat][f"{k}"] = 0

    for prompt in tqdm(gt):
        prompt_cat = PROMPT_TO_CAT[prompt]

        gt_videos = list(gt[prompt].keys())
        
        total["all"] += 1
        total[prompt_cat] += 1

        videos = pred[prompt]["videos"]
        scores = pred[prompt]["scores"]

        scores, videos = zip(*sorted(zip(scores, videos)))
        scores = scores[::-1]
        videos = videos[::-1]

        for k in ks:
            recall_k_videos = videos[:k]
            
            for v in recall_k_videos:
                if v in gt_videos:
                    count["all"][f"{k}"] += 1
                    count[prompt_cat][f"{k}"] += 1
                    break

    results = {}

    for cat in PROMPT_CATEGORIES:
        if total[cat] > 0:
            results[cat] = {}
            results[cat]["total_prompt_count"] = total[cat]

            for k in ks:
                results[cat][f"R@{k}"] = (count[cat][f"{k}"] / total[cat]) * 100

    return results

def evaluate_moment_retrieval(gt_data, pred_data):
    gt, pred = load_data(gt_data, pred_data)

    score_dict = { }
    for cat in PROMPT_CATEGORIES:
        score_dict[cat] = { }

    tIoUs = [ 0.5, 0.7 ]
    for tIoU in tIoUs:
        scores = {}
            
        for cat in PROMPT_CATEGORIES:
            scores[cat] = []

        for prompt in tqdm(gt):
            prompt_cat = PROMPT_TO_CAT[prompt]
            
            for video in gt[prompt]:
                if gt[prompt][video]["clip"]:
                    gt_bounds = gt[prompt][video]["bounds"]
                    pred_bounds = pred[prompt][video]["bounds"]

                    iou = compute_iou(gt_bounds, pred_bounds)

                    if iou < tIoU:
                        score = 0
                    else:
                        score = 1

                    scores["all"].append(score)
                    scores[prompt_cat].append(score)

        
        for cat in PROMPT_CATEGORIES:
            if len(scores[cat]) > 0:
                score_dict[cat]["total_videos"] = len(scores[cat])
                score_dict[cat][f"R@{tIoU}"] = np.mean(scores[cat]) * 100

    return score_dict

def compute_step_bound_scores(gt_data, pred_data):
    gt, pred = load_data(gt_data, pred_data)

    results = {}
    
    for cat in PROMPT_CATEGORIES:
        results[cat] = {}

        results[cat]["recall"] = {}
        results[cat]["precision"] = {}
    
    for tiou in [ 0.5, 0.7 ]:
        
        recall = {}
        precision = {}
        ious = {}

        for cat in PROMPT_CATEGORIES:
            recall[cat] = []
            precision[cat] = []
            ious[cat] = []

        for i, video in tqdm(enumerate(gt), total=len(gt)):
            video_cat = VIDEOS_TO_CAT[video]

            best_recall = 0
            best_precision = 0
            
            ref_set_covered = set([])
            pred_set_covered = set([])

            refs = gt[video]["bounds"]
            preds = pred[video]["bounds"]

            for pred_i, pred_x in enumerate(preds):
                local_ious = []
                
                for ref_i, gt_x in enumerate(refs):
                    iu = compute_iou(pred_x, gt_x)

                    local_ious.append(iu)
                    if iu > tiou:
                        ref_set_covered.add(ref_i)
                        pred_set_covered.add(pred_i)

                ious[video_cat].append(max(local_ious))
                ious["all"].append(max(local_ious))

            new_precision = float(len(pred_set_covered)) / (pred_i + 1) 
            best_precision = max(best_precision, new_precision)

            new_recall = float(len(ref_set_covered)) / len(refs)
            best_recall = max(best_recall, new_recall)

            recall[video_cat].append(best_recall)
            precision[video_cat].append(best_precision)
            recall["all"].append(best_recall)
            precision["all"].append(best_precision)

        for cat in PROMPT_CATEGORIES:
            if len(recall[cat]) > 0:
                results[cat]["recall"][f"{tiou}"] = sum(recall[cat]) / len(recall[cat]) * 100
                results[cat]["precision"][f"{tiou}"] = sum(precision[cat]) / len(precision[cat]) * 100
                results[cat]["total"] = len(recall[cat])
            
    return results

def evaluate_moment_summarization(gt_data, pred_data, gpu_device: int):
    gt, pred = load_data(gt_data, pred_data)

    all_results = {}
    bert_scores = []

    
    from allennlp_models import pretrained
    predictor = pretrained.load_predictor(
        "pair-classification-decomposable-attention-elmo",
        cuda_device=gpu_device
        )
    print("Loaded Entailment evaluation model")

    if gpu_device != -1 and args.frame_dir != "None":
        import clip
        model, preprocess = clip.load("ViT-B/32", device=f"cuda:{gpu_device}")
        from glob import glob
        import torch
        from PIL import Image
        print("Loaded CLIP model")

    for cat in PROMPT_CATEGORIES:
        refs = []
        cands = []

        total_videos = 0

        entailment_scores = [ 0, 0, 0 ]
        total_entailment_count = 0

        k = 0
        clip_scores = []
        entailment_list = []
        for video in tqdm(gt):
            video_cat = VIDEOS_TO_CAT[video]
            vid_clip_scores = []

            if cat == video_cat or cat == 'all':
                total_videos += 1

                for i, d in enumerate(gt[video]["captions"]):

                    gt_sent = d["sentence"].lower()
                    cand = pred[video]["captions"][i]["sentence"].lower()

                    if gpu_device != -1 and args.frame_dir != "None":
                        frames = glob(f"{args.frame_dir}/{video}/*.jpg")
                        frames.sort(key=lambda a : int(a.split("_")[-1].replace(".jpg", "")))

                        skip = False
                        if d["start"] >= len(frames) or d["end"] >= len(frames):
                            skip = True

                        if not skip:
                            text = clip.tokenize([cand]).to(f"cuda:{gpu_device}")
                            idxes = np.linspace(d["start"], min(d["end"], len(frames))-1, 4).astype(int)

                            frames = np.array(frames)

                            img_features = [ ]
                            for frame in frames[idxes]:
                                image = preprocess(Image.open(frame)).cpu()
                                img_features.append(image)

                            img_features = torch.stack(img_features)

                            with torch.no_grad():
                                image_features = model.encode_image(img_features.to(f"cuda:{gpu_device}"))
                                text_features = model.encode_text(text)
                                
                                image_features /= image_features.norm(dim=-1, keepdim=True)
                                text_features /= text_features.norm(dim=-1, keepdim=True)

                                dot_score = image_features @ text_features.T

                                score = torch.mean(dot_score)

                                vid_clip_scores.append(float(score.cpu()))
    
                    k += 1

                    refs.append(gt_sent)
                    cands.append(cand)

                    x = predictor.predict(
                        premise=gt_sent,
                        hypothesis=cand
                    )
                    entail_idx = np.argmax(x["label_probs"])
                    if entail_idx == 0:
                        entailment_list.append(1)
                    else:
                        entailment_list.append(0)

                    entailment_scores[entail_idx] += 1
                    total_entailment_count += 1

            clip_scores.extend(vid_clip_scores)

        if len(refs) == 0 or len(cands) == 0:
            continue

        print("Computing BERTScore...")
        from bert_score import score
        p, r, f = score(cands, refs, lang='en', verbose=True,
                        device=f"cuda:{gpu_device}"
                        )

        print("Computing COCO Eval metrics...")
        evaluator = language_evaluation.CocoEvaluator()
        coco_results = evaluator.run_evaluation(cands, refs)
        
        if len(clip_scores) == 0:
            clip_scores = [ 0 ]

        results = {
            "CLIPScore": np.average(clip_scores),
            "BERTScore_F1": f.mean().item(),
            "Total": total_videos,
            "Entailment": (entailment_scores[0] / total_entailment_count) * 100,
            "Contradiction": (entailment_scores[1] / total_entailment_count) * 100,
            "Netural": (entailment_scores[2] / total_entailment_count) * 100
        }

        for metric in coco_results:
            results[metric] = coco_results[metric] * 100

        all_results[cat] = results
        
    return all_results

def NMS(boxes, overlapThresh=0):
	if len(boxes) == 0:
		return []
    
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")

	pick = []

	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
    
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
    
	while len(idxs) > 0:
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
        
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		
		overlap = (w * h) / area[idxs[:last]]
		
		idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
	
	return boxes[pick]

def preprocess_moment_bounds(gt_data, pred_data):
    gt, pred = load_data(gt_data, pred_data)

    for i, video in tqdm(enumerate(pred), total=len(pred)):
        bounds = pred[video]["bounds"]
        gt_bounds = gt[video]["bounds"]
        min_x = gt_bounds[0][0]
        max_x = gt_bounds[-1][1]

        bounds = [bound for bound in bounds if (bound[0] > min_x and bound[1] < max_x) ]

        x1 = [ ]
        x2 = [ ]
        y1 = [ 0 ] * len(bounds)
        y2 = [ 1 ] * len(bounds)

        for bound in bounds:
            x1.append(bound[0])
            x2.append(bound[1])

        boxes = np.zeros((len(bounds), 4))
        boxes[:, 0] = x1
        boxes[:, 1] = y1
        boxes[:, 2] = x2
        boxes[:, 3] = y2
        
        boxes = NMS(boxes)

        if len(boxes) > 0:
            x1 = boxes[:, 0]
            x2 = boxes[:, 2]

            bounds = []
            for i in range(len(x1)):
                bounds.append([ x1[i], x2[i] ])

            bounds.sort(key=lambda x: x[0])
            new_bounds = []

            if bounds[0][0] > min_x:
                new_bounds.append([min_x, bounds[0][0]])
            
            for i in range(0, len(bounds)):
                new_bounds.append(bounds[i])
                if (i+1) < len(bounds):
                    new_bounds.append([bounds[i][1], bounds[i+1][0]])
            
            if new_bounds[-1][1] < max_x:
                new_bounds.append([new_bounds[-1][1], max_x])
        else:
            new_bounds = [ [min_x, max_x] ]

        pred[video]["bounds"] = new_bounds

    return pred

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiment', add_help=False)

    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--gt_data', type=str, required=False)
    parser.add_argument('--pred_data', type=str, required=True)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--print_per_category', action='store_true')
    parser.add_argument('--help', action='store_true')
    parser.add_argument('--preprocess_moment_bounds', action='store_true')
    parser.add_argument('--replace_pred_moment_bounds', action='store_true')
    parser.add_argument('--frame_dir', type=str, default="None")
    args = parser.parse_args()

    print(args)

    if args.preprocess_moment_bounds:
        if args.gt_data is None:
            args.gt_data = './data/evaluation/formatted_moment_evaluation_gt.json'
                
        new_pred = preprocess_moment_bounds(args.gt_data, args.pred_data)
            
        if args.replace_pred_moment_bounds:
            assert isinstance(args.pred_data, str), "You must provide a path to the source file"

            with open(args.pred_data, 'w') as f:
                json.dump(new_pred, f)

        args.pred_data = new_pred

    PROMPT_CATEGORIES = set()
    PROMPT_TO_CAT = { }
    VIDEOS_TO_CAT = { }

    category_path = './data/evaluation/categories.json'
    
    with open(category_path, 'r') as f:
        data = json.load(f)
        PROMPT_TO_CAT = data["prompt_to_cat"]
        VIDEOS_TO_CAT = data["video_to_cat"]

        for p in PROMPT_TO_CAT:
            PROMPT_CATEGORIES.add(PROMPT_TO_CAT[p])
        
        for v in VIDEOS_TO_CAT:
            PROMPT_CATEGORIES.add(VIDEOS_TO_CAT[v])
    
    PROMPT_CATEGORIES = list(PROMPT_CATEGORIES)
    PROMPT_CATEGORIES.append("all")

    if (args.help):
        print("Please see the 'examples_for_evaluation_folder' for input examples")
    else:
        if args.task == "video_retrieval":
            if args.gt_data is None:
                args.gt_data = './data/splits/all_data_test.json'

            result = evaluate_video_retrieval(args.gt_data, args.pred_data)

        elif args.task == "moment_retrieval":
            if args.gt_data is None:
                args.gt_data = './data/splits/all_data_test.json'

            result = evaluate_moment_retrieval(args.gt_data, args.pred_data)

        elif args.task == "moment_segmentation":
            if args.gt_data is None:
                args.gt_data = './data/evaluation/formatted_moment_evaluation_gt.json'


            result = compute_step_bound_scores(args.gt_data, args.pred_data)

        elif args.task == "step_captioning":
            if args.gt_data is None:
                args.gt_data = './data/evaluation/formatted_moment_evaluation_gt.json'

            if not (args.print_per_category):
                PROMPT_CATEGORIES = [ "all" ]

            result = evaluate_moment_summarization(args.gt_data, args.pred_data, args.device)
        else:
            result = { "all": {} }

    
        if not (args.print_per_category):
            print(result["all"])
        else:
            print(result)