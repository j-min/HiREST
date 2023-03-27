import whisper
import json
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='small.en')
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--asr_dir", type=str, required=True)

    args = parser.parse_args()
    print(args)

    audio_dir = Path(args.audio_dir)
    ASR_dir = Path(args.asr_dir)


    # Model = 'small.en' #@param ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large']
    # Model = 'large.en' #@param ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large']

    Model = args.model

    whisper_model = whisper.load_model(Model)

    whisper_model = whisper_model.to('cuda')
    whisper_model.eval()
    print('Model loaded: ', Model, "at device", whisper_model.device)

    #@markdown ### **Behavior control**
    #@markdown ---
    language = "English" #@param ['Auto detection', 'Afrikaans', 'Albanian', 'Amharic', 'Arabic', 'Armenian', 'Assamese', 'Azerbaijani', 'Bashkir', 'Basque', 'Belarusian', 'Bengali', 'Bosnian', 'Breton', 'Bulgarian', 'Burmese', 'Castilian', 'Catalan', 'Chinese', 'Croatian', 'Czech', 'Danish', 'Dutch', 'English', 'Estonian', 'Faroese', 'Finnish', 'Flemish', 'French', 'Galician', 'Georgian', 'German', 'Greek', 'Gujarati', 'Haitian', 'Haitian Creole', 'Hausa', 'Hawaiian', 'Hebrew', 'Hindi', 'Hungarian', 'Icelandic', 'Indonesian', 'Italian', 'Japanese', 'Javanese', 'Kannada', 'Kazakh', 'Khmer', 'Korean', 'Lao', 'Latin', 'Latvian', 'Letzeburgesch', 'Lingala', 'Lithuanian', 'Luxembourgish', 'Macedonian', 'Malagasy', 'Malay', 'Malayalam', 'Maltese', 'Maori', 'Marathi', 'Moldavian', 'Moldovan', 'Mongolian', 'Myanmar', 'Nepali', 'Norwegian', 'Nynorsk', 'Occitan', 'Panjabi', 'Pashto', 'Persian', 'Polish', 'Portuguese', 'Punjabi', 'Pushto', 'Romanian', 'Russian', 'Sanskrit', 'Serbian', 'Shona', 'Sindhi', 'Sinhala', 'Sinhalese', 'Slovak', 'Slovenian', 'Somali', 'Spanish', 'Sundanese', 'Swahili', 'Swedish', 'Tagalog', 'Tajik', 'Tamil', 'Tatar', 'Telugu', 'Thai', 'Tibetan', 'Turkish', 'Turkmen', 'Ukrainian', 'Urdu', 'Uzbek', 'Valencian', 'Vietnamese', 'Welsh', 'Yiddish', 'Yoruba']
    # verbose = 'Live transcription' #@param ['Live transcription', 'Progress bar', 'None']
    verbose = 'None'
    output_type = 'All' #@param ['All', '.txt', '.vtt', '.srt']
    task = 'transcribe' #@param ['transcribe', 'translate']
    temperature = 0.15 #@param {type:"slider", min:0, max:1, step:0.05}
    temperature_increment_on_fallback = 0.2 #@param {type:"slider", min:0, max:1, step:0.05}
    best_of = 5 #@param {type:"integer"}
    beam_size = 5 #@param {type:"integer"}
    patience = 1.0 #@param {type:"number"}
    length_penalty = -0.05 #@param {type:"slider", min:-0.05, max:1, step:0.05}
    suppress_tokens = "-1" #@param {type:"string"}
    initial_prompt = "" #@param {type:"string"}
    condition_on_previous_text = True #@param {type:"boolean"}
    fp16 = True #@param {type:"boolean"}
    compression_ratio_threshold = 2.4 #@param {type:"number"}
    logprob_threshold = -1.0 #@param {type:"number"}
    no_speech_threshold = 0.6 #@param {type:"slider", min:-0.0, max:1, step:0.05}

    verbose_lut = {
        'Live transcription': True,
        'Progress bar': False,
        'None': None
    }

    whisper_args = dict(
        language = (None if language == "Auto detection" else language),
        verbose = verbose_lut[verbose],
        task = task,
        temperature = temperature,
        temperature_increment_on_fallback = temperature_increment_on_fallback,
        best_of = best_of,
        beam_size = beam_size,
        patience=patience,
        length_penalty=(length_penalty if length_penalty>=0.0 else None),
        suppress_tokens=suppress_tokens,
        initial_prompt=(None if not initial_prompt else initial_prompt),
        condition_on_previous_text=condition_on_previous_text,
        fp16=fp16,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold
    )

    temperature = whisper_args.pop("temperature")
    temperature_increment_on_fallback = whisper_args.pop("temperature_increment_on_fallback")
    if temperature_increment_on_fallback is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback))
    else:
        temperature = [temperature]

    all_audio_paths = list(audio_dir.glob('*.wav'))

    audio_paths = all_audio_paths

    for audio_path in tqdm(audio_paths):

        video_transcription = whisper.transcribe(
            whisper_model,
            str(audio_path),
            temperature=temperature,
            **whisper_args
        )

        ASR_path = ASR_dir / (audio_path.stem + '.srt')

        with open(ASR_path, 'w', encoding='utf-8') as f:
            whisper.utils.write_srt(video_transcription['segments'], file=f)
