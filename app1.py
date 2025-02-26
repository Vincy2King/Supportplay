#!/user/bin/env python3
# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import cv2
import math
from moviepy.editor import VideoFileClip, AudioFileClip
app = Flask(__name__)
import sys
openai_model_config = {
    "config_name": "gpt-4o_config",
    "model_type": "openai_chat",
    "model_name": "gpt-4o-mini",
    "api_key": key,
}

from agentscope.agents import DialogAgent, UserAgent
import agentscope
from agentscope.message import Msg
import json, re, time
from datetime import datetime
import time
import json
import re
import random
import yaml
import argparse
import torch
import os
from munch import Munch
import numpy as np
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from StyleTTS2.styletts2 import StyleTTS2
# import IPython.display as ipd
from scipy.io import wavfile
from EAT.demo import EAT
from EAT.preprocess.deepspeech_features.extract_ds_features import extract_features
# 获取当前时间的时间戳

turns = 0

emotion_aligned = {
    "suprised": "sur",
    "excited": "hap",
    "angry": "ang",
    "proud": "hap",
    "sad": "sad",
    "annoyed": "ang",
    "grateful": "hap",
    "lonely": "sad",
    "afraid": "fea",
    "terrified": "fea",
    "guilty": "sad",
    "impressed": "sur",
    "disgusted": "dis",
    "hopeful": "hap",
    "confident": "neu",
    "furious": "ang",
    "anxious": "sad",
    "anticipating": "hap",
    "joyful": "hap",
    "nostalgic": "sad",
    "disappointed": "sad",
    "prepared": "neu",
    "jealous": "ang",
    "content": "hap",
    "devastated": "sur",
    "embarrassed": "neu",
    "caring": "hap",
    "sentimental": "sad",
    "trusting": "neu",
    "ashamed": "neu",
    "apprehensive": "fea",
    "faithful": "neu",
}

def extract_first_integer(s):
    match = re.search(r'\d+', s)
    if match:
        return int(match.group())
    return None

def read_npy(npy_path):
    dialogue_list = []
    data = np.load(npy_path, allow_pickle=True)
    for i, dia in enumerate(data):
        history = ""
        if len(data) > 1:
            for dia_num in range(len(data[i]) - 1):
                if dia_num % 2 == 0:
                    utt = '[user]' + dia[dia_num]
                else:
                    utt = '[agent]' + dia[dia_num]
                history += utt
        else:
            history = "None"
        user_query = data[i][-1]
        dialogue_list.append(f"User Query:{user_query} \n Conversation History:{history}\n ")
    return dialogue_list


# Load model configs
agentscope.init(model_configs=openai_model_config)

# Create a dialog agent and a user agent
character_agent = DialogAgent(name="assistant",
                              model_config_name="gpt-4o_config",
                              sys_prompt="You're a helpful assistant")
# You are a psychological counselor with twenty years of experience and are good at \
# reconstructing psychological counseling scenes.
supporter_agent = DialogAgent(name="supporter",
                              model_config_name="gpt-4o_config",
                              sys_prompt="You're a helpful assistant")
emotion_agent = DialogAgent(name="emotion_assistant",
                            model_config_name="gpt-4o_config",
                            sys_prompt="You're a helpful assistant")

score_agent = DialogAgent(name="score_assistance",
                          model_config_name="gpt-4o_config",
                          sys_prompt="You're a helpful assistant")
# user_agent = UserAgent()

num = 0
gn = 1
all_dict = {}
all_supporter_memory = []
seeker_dialog = []
supporter_memory = ''
# all_tmp_memory = []
with open('results_oral_test.json','r',encoding='utf-8') as f:
    all_tmp_memorys = json.load(f)

'''
strategy_list = ['Question', 'Affirmation and Reassurance', 'Others', 'Information',
                 'Self-disclosure', 'Providing Suggestions', 'Restatement or Paraphrasing',
                 'Reflection of feelings']
'''
strategy_list = ['Reflective Statements','Clarification','Emotional Validation','Empathetic Statements',
'Affirmation','Offer Hope','Avoid Judgment And Criticism','Suggest Options',
'Collaborative Planning','Provide Different Perspectives','Reframe Negative Thoughts',
'Share Information','Normalize Experiences','Promote Self-Care Practices',
'Stress Management','Others']
emotion_list = ['anxiety', 'anger', 'fear', 'depression', 'sadness', 'disgust', 'shame', 'nervousness', 'pain', 'jealousy', 'guilt']

score_dict = {
    "Informativeness":"",
    "Understanding":"",
    "Helpfulness":"",
    "Consistency":"",
    "Coherence":""
}
score_list = [
 "Informativeness measures how well the individual seeking support articulates their emotional challenges.",
 "Understanding gauges the supporter's grasp of the individual's experiences and emotions.",
 "Helpfulness evaluates the effectiveness of the supporter's efforts in mitigating the individual's emotional distress.",
 "Consistency ensures participants consistently adhere to their roles and exhibit non-contradictory behavior",
 "Coherence checks if conversations have seamless topic transitions.",
]

x = None

def get_message(content, mem, name):
    message = Msg(
        name=name,
        content=content,
        role="user",
        mem = mem
    )
    # message['mem'] = mem
    return message


def replace_quotes(string):
    parts = string.split("'", 3)
    new_string = '"'.join(parts[:3]) + parts[3]
    parts = new_string.rsplit("'", 3)
    new_string = parts[0] + '"'.join(parts[1:])
    return new_string

def demo(input,name,sex,education,hobby, turns):
    while True:
        global all_tmp_memorys, initial_emo, emotion_list, seeker_dialog
        similar_character = []
        tmp_memory = {}
        data = {
            'name':name,
            'sex':sex,
            'education':education,
            'hobby':hobby
        }
        if data is not None:
            for item, value in all_tmp_memorys.items():

                similar, similar_memory = character_agent(get_message(
                    '[character1]:' + str(data) + '\n[character2]:' + str(value['character']) +
                    '\nPlease give a integer score from 1-10 to calculate the simiary between [character1] '
                    'and [character2]', 'add', 'character'))
                print('similar:', similar['content'])
                similar = extract_first_integer(similar['content'])
                # if similar!=None:

                similar_character.append(int(similar))
                if similar > 5:
                    tmp_memory[item] = value
            all_tmp_memorys = tmp_memory

        long_memory = []
        for item, value in all_tmp_memorys.items():
            long_memory.append(value['all_memory'])

        all_supporter_memory = []

        if turns == 0:
            emo, emo_memory = emotion_agent(get_message(input + '\n Please give the emotion category ' + str(
                emotion_list) +
                                                        ' and emotion degree of the above sentence from 1 to 5 score and fill the following json:\n' + str(
                initial_emotion_dict), 'add', 'emotion_assistant'))

            begin = emo['content'].replace('\n', '').find('{')
            end = emo['content'].replace('\n', '').find('}')
            print('emo:',emo['content'])
            try:
                emo_dict = eval(emo['content'].replace('\n', '')[begin:end + 1])
            except:
                continue
            initial_emo = emo_dict['emotion category']
        else:
            emo, emo_memory = emotion_agent(get_message(input + '\n emotion category:' + initial_emo + '\n Based on the initial emotion category, '
                                                                                                                'please give the emotion of the above sentence from 1 to 5 score and '
                                                                                                                'fill the following json:\n' + str(
                emotion_dict), 'add', 'emotion_assistant'))

        seeker_res = {
            "content": input,
            "emotion": emo['content'],
        }
        supporter_dialog = []
        if (turns + 1) % 3 == 0:
            question_res, supporter_memory = supporter_agent(
                get_message(
                    'This is seeker\'s profile:\n' + str(data) +
                    '\n\nThis is the utterance from seeker' + seeker_res['content'] + '\n\n'
                                                                                      '(1) Retrieve the related memories;'
                                                                                      '(2) Given only the information above, what are 5 most salient highlevel questions about seeker\'s profile and suppoert strategies we can answer about the subjects in the statements? '
                                                                                      'Please fill the following json:\n' + str(
                        memory_dict), 'delete', 'supporter'))


            print('question_res:', question_res)
            print('supporter_memory 2:', supporter_memory)
            begin = question_res['content'].replace('\n', '').find('{')
            end = question_res['content'].replace('\n', '').find('}')
            res_dict = question_res['content'].replace('\n', '')[begin:end + 1].replace("'", '"')
            res_dict = res_dict.replace('"s', "'s").replace('you"re', "you're").replace('you"ve', "you've")
            print('res_dict:', res_dict)
            try:
                insight_mem_res, supporter_memory = supporter_agent(
                    get_message(
                        str(eval(res_dict)['questions']) + '\nBased on the above question to retrieve insight memories and '
                                                           'fill the following json:\n' + str(question_memory_dict), 'delete',
                        'supporter'), long_memory)
            except:
                continue
            print('insight_mem_res:', insight_mem_res)
            print('supporter_memory 3:', supporter_memory)
            # based on the related memores, obtain insight and save to the memory
            insight_res, supporter_memory = supporter_agent(
                get_message(insight_mem_res[
                                'content'] + '\nWhat 5 high-level insights can you infer from the above statements? and '
                                             'fill the following json:\n' + str(insight_dict), 'add', 'supporter'),
                long_memory)
            print('insight_res:', insight_res)
            print('supporter_memory 4:', supporter_memory)

            insight = insight_res
            all_supporter_memory.append(insight)

        if turns > 0:
            useful_memory = []
            relevance_list = []
            pt_list = []
            print('all:',type(all_supporter_memory),all_supporter_memory)
            for memory in all_supporter_memory:
                relevance_result,_ = supporter_agent(get_message('[Memory]:'+memory['content']+'\n[Utterance]:'+seeker_res['content']+
                                                '\nPlase give a number from 0-10 to evaluate the similarity between [Memory] and [Utterance]','delete','supporter'))
                print('relevance_result:', relevance_result)
                relevance = int(re.search(r'\d+', relevance_result['content']).group())
                now = datetime.now()
                time1 = now.strftime("%Y-%m-%d %H:%M:%S")
                time2 = memory['timestamp']
                time_format = "%Y-%m-%d %H:%M:%S"
                time1 = datetime.strptime(time1, time_format)
                time2 = datetime.strptime(time2, time_format)
                print(time1, time2, time1 - time2)
                score = 1/((time1-time2).total_seconds()) * relevance
                t = (time1-time2).total_seconds()
                global gn
                gn = gn + (1-math.exp(-t))/(1+math.exp(-t))
                a = 1/gn
                pt = 1 - math.exp(-relevance*math.exp(-a*t))
                pt_list.append(pt)
                relevance_list.append(score)
                if score>0.05 and pt > 0.005:
                    useful_memory.append(memory)
            print('pt_list:',pt_list)
            print('score:',relevance_list)
            print('use:',useful_memory)
            all_supporter_memory = useful_memory

            supporter_res,supporter_memory = supporter_agent(
                    get_message('This is seeker\'s profile:\n' + str(data) +
                        '\n\nThis is the seeker emotion and preference for the historical dialog:\n' +
                                emo['content'] + '\n\n This is the utterance from seeker' + seeker_res['content'] + '\n\n'
                                'Here are the strategies that you can use to respond: ' + str(strategy_list) + '\n\n'
                                # 'Here are some experiment you can refer to: '+insight+'\n\n'
                                + '(1) Based on your memory, find the corresponding dialogues.\n'
                                  '(2) Based on your memory, summarize the effective experience of different support methods for different types of seekers by point\n'
                                  '(3) Learn from the response strategy for different user.\n'
                                  '(4) Use no more than 2 sentences in colloquial expression in English to support user. (少用我明白我理解)\n'
                                  'Please fill the following json:\n' + str(supporter_dict), 'add', 'supporter'),all_supporter_memory)


        else:
            supporter_res,supporter_memory = supporter_agent(
                get_message('This is seeker\'s profile:\n' + str(data) +
                                '\n\nThis is the utterance from seeker:' + seeker_res['content'] + '\n\n' +
                                'Here are the strategies that you can use to respond: ' + str(strategy_list) + '\n\n' 
                                                                                                               # 'Here are some experiment you can refer to: '+insight+'\n\n'
                                                                                                               '(1) Based on your memory, find the corresponding dialogues.\n'
                                                                                                               '(2) Based on your memory, summarize the effective experience of different support methods for different types of seekers by point\n'
                                '(3) Learn from the response strategy for different user.\n'
                                                                                                               '(4) Output no more than 2 sentences in colloquial expression in English to support user. (少用我明白我理解)\n'
                                                                                                               'Please fill the following json:\n' + str(
                        supporter_dict), 'add', 'supporter'),all_supporter_memory)

        begin = supporter_res['content'].replace('\n', '').find('{')
        end = supporter_res['content'].replace('\n', '').find('}')
        print(begin, end)
        res_dict = supporter_res['content'].replace('\n', '')[begin:end + 1].replace('\n', '') \
            .replace("'similar dialog in memory': '", '"similar dialog in memory": "') \
            .replace("'summaried experimence': '", '"summaried experimence": "') \
            .replace("'strategy': '", '"strategy": "') \
            .replace("'response': '", '"response": "') \
            .replace("',", '",') \
            .replace("'}", '"}')
        supporter_dialog.append('supporter:' + res_dict)

        try:
            res_dict = eval(res_dict)
        except:
            continue
        print(res_dict['response'])
        score_res, score_memory = score_agent(
            get_message('Seeker:' + seeker_res['content'] +
                        '\nSupporter:' + res_dict['response'] + '\n' +
                        'Here are the score standards:' + '\n'.join(score_list) + '\n' +
                        'Based on the utterance between seeker and supporter and fill the following json for 1-5 score:\n' + str(
                score_dict), 'add', 'score'))
        print('score res:', score_res)
        text = score_res['content']
        start_index = text.find("{")
        end_index = text.rfind("}") + 1
        json_str = text[start_index:end_index].replace('\n', '') \
            .replace("'Informativeness': '", '"Informativeness": "') \
            .replace("'Understanding': '", '"Understanding": "') \
            .replace("'Helpfulness': '", '"Helpfulness": "') \
            .replace("'Consistency': '", '"Consistency": "') \
            .replace("'Coherence': '", '"Coherence": "') \
            .replace("'}", '"}') \
            .replace("',", '",')
        try:
            score_content = eval(json_str)
        except:
            continue
        print('score content:', score_content)

        content = 'This is seeker\'s profile:\n' + str(data) + '\n\n'
        content += 'This is the utterance from seeker:' + seeker_res['content'] + '\n\n'
        content += 'This is seeker\' emotion:' + seeker_res['emotion'] + '\n\n'
        content += 'This is supporter response:' + supporter_res['content'] + '\n\n'
        content += 'This is the score of supporter response toward seeker:' + json_str

        temp_mem_dict = Msg(
            id=supporter_res['id'],
            timestamp=supporter_res['timestamp'],
            name=supporter_res['name'],
            content=content,
            role=supporter_res['role'],
            url=supporter_res['url'],
            metadata=supporter_res['metadata']
        )

        all_supporter_memory.append(temp_mem_dict)

        emo, emo_memory = emotion_agent(get_message(seeker_res[
                                                        'content'] + '\n emotion category:' + initial_emo + '\n Based on the initial emotion category, '
                                                                                                            'please give the emotion of the above sentence from 1 to 5 score and '
                                                                                                            'fill the following json:\n' + str(
            emotion_dict), 'add', 'emotion_assistant'))
        # turns += 1
        seeker_res['emotion'] = emo['content']
        seeker_dialog.append('seeker:' + str(seeker_res))

        all_dict[str(num)] = {
            "character": data,
            "seeker": seeker_dialog,
            "supporter": supporter_dialog,
            "all_memory": all_supporter_memory
        }

        wav_save_path = args.wav_save_path + f'test{name}_{turns}'
        mp4_save_path = args.mp4_save_path + f'test{name}_{turns}'
        for folder_path in [wav_save_path, mp4_save_path]:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Folder '{folder_path}' created.")
                continue_outer = False
            else:
                print(f"Folder '{folder_path}' already exists.")
                continue_outer = True
        # if continue_outer:
        #     continue
        if not os.path.exists(wav_save_path + '/deepfeature32'):
            os.makedirs(wav_save_path + '/deepfeature32')

        # ===========begin generate===========
        agent_gender = 'Female'  # data["Agent Gender"]  # .strip('"')
        agent_age = 'Teenagers'  # data["Agent Age"]  # .strip('"')
        agent_timbre_tone = 'Energetic'  # data["Agent Timbre and Tone"]  # .strip('"')
        # if agent_timbre_tone not in ['bright','clear','husky','low_pitched','melodious','soft','warm']:
        #     continue
        empathetic_response = res_dict['response']
        emotional_response = 'Excited'
        emotion_type = "neu"
        # TTS for response text
        tts = StyleTTS2()
        if agent_gender == 'Female':
            wav_file = "StyleTTS2/Demo/reference_audio/W/soft.wav"
        elif agent_gender == 'Male':
            wav_file = "StyleTTS2/Demo/reference_audio/M/soft.wav"
        result_name = wav_file.split('/')[-1]
        start = time.time()
        noise = torch.randn(1, 1, 256).to(args.device)
        ref_s = tts.compute_style(wav_file)
        wav = tts.inference(empathetic_response, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1)
        rtf = (time.time() - start) / (len(wav) / 24000)
        print(f"RTF = {rtf:5f}")
        scaled_data = np.int16(wav * 32767)
        wav_file_path = wav_save_path + '/' + result_name
        wavfile.write(wav_file_path, 24000, scaled_data)

        # extract wav deepspeech features
        extract_features(wav_save_path, wav_save_path + '/deepfeature32')
        # wav2talkingface
        eat = EAT(root_wav=wav_save_path)
        video_path, audio_path = eat.tf_generate(agent_age, agent_gender, emotion_type, save_dir=mp4_save_path)
        print('save end',video_path, audio_path)
        # 并截取第一帧作为图片，返回视频和图片地址
        cap = cv2.VideoCapture(video_path)
        # 读取第一帧
        img_path = 'static/assets/img/'+mp4_save_path.split('/')[-1].split('.')[0]+'.jpg'

        success, frame = cap.read()
        if success:
            # 保存第一帧为图片
            cv2.imwrite(img_path, frame)
            print("第一帧已保存为 first_frame.jpg")
        # 合并音频和视频
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
        video_with_audio = video.set_audio(audio)
        save_mp4_path = 'static/assets/mp4/'+mp4_save_path.split('/')[-1].split('.')[0]+'_output_video.mp4'
        video_with_audio.write_videofile(save_mp4_path, codec="libx264", audio_codec="aac")

        save_mp4_path, img_path = '',''
        return save_mp4_path,img_path, res_dict['response']

@app.route('/logout', methods=['POST'])
def logout():
    return render_template('login.html')

@app.route('/', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        # print('personal_info')
        temp_name = request.form.get('name')  # form取post方式参数
        temp_age = request.form.get('age')
        temp_sex = request.form.get('sex')
        temp_education = request.form.get('education')
        temp_hobby = request.form.get('hobby')  # getlist取一键多值类型的参数
        temp_nickname = request.form.get('nickname')
        temp_occupation = request.form.get('occupation')
        temp_appearance = request.form.get('appearance')
        name = temp_name
        age = temp_age
        sex = temp_sex
        education = temp_education
        hobby = temp_hobby
        nickname = temp_nickname
        occupation = temp_occupation
        appearance = temp_appearance
        print('name:',temp_name,temp_age,temp_sex,temp_education,temp_hobby)
        user = {
            'name':temp_name,
            'age':temp_age,
            'sex':temp_sex,
            'education':temp_education,
            'hobby':temp_hobby,
            'nickname':nickname,
            'occupation':temp_occupation,
            'appearance':appearance
        }
        return render_template('index.html',user=user)
    elif request.method=='GET':
        return render_template('login.html')

# 显示主页
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/get_input', methods=['POST'])
def get_input():
    input_data = request.form.get('input_value')
    print(f"收到的输入内容: {input_data}")
    return jsonify({'status': 'success', 'received_data': input_data})

# 处理提交的数据
@app.route('/submit_message', methods=['GET', 'POST'])
def submit_message():
    print('here')
    user_input = request.form.get('chat-input')
    print(f"用户输入的消息: {user_input}")

    return render_template('index.html')


@app.route('/process', methods=['POST'])
def get_list_items():
    print('process')
    data = request.json

    global turns
    people = '2'
    turns+=1
    txt = [
        "It sounds really tough to feel that way constantly. Just remember, it's okay to ask for help and take a breather when you need it.",
        "It can really get to you when everything feels like so much. Just know that many people go through the same thing, and you're definitely not alone in this."
    ]
    mp4_img = ''
    mp4_file = 'static/assets/mp4/'+people+'/expressionSet1_'+str(turns)+'.mp4'
    import time
    time.sleep(5)

    # mp4_file = 'static/assets/mp4/age25_001_neu_soft.mp4'
    return jsonify({'mp4_file': mp4_file,'mp4_img':mp4_img,'response':txt[turns-1]})

if __name__ == '__main__':
    # initialization
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(0)
    np.random.seed(0)

    # Argument Parser Setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="ChatGLM3",
                        help="The directory of the model")
    parser.add_argument("--tokenizer", type=str, default="ChatGLM3", help="Tokenizer path")
    parser.add_argument("--LoRA", type=str, default=True, help="use lora or not")
    parser.add_argument("--lora-path", type=str, default='chatglm/checkpoint-24000/pytorch_model.pt',
                        help="Path to the LoRA model checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum new tokens for generation")
    parser.add_argument("--lora-alpha", type=float, default=32, help="LoRA alpha")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA r")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--wav_save_path", type=str, default="TTS_results/ED_test/")
    parser.add_argument("--mp4_save_path", type=str, default="MP4_results/ED_test/")
    parser.add_argument("--driven_video", type=str, default="EAT/demo/video_processed/template")
    args = parser.parse_args()

    # with open('results.json','w',encoding='utf-8') as f:
    num = 0
    all_dict = {}
    all_supporter_memory = []
    supporter_memory = ''
    # num = 0
    initial_emotion_dict = {
        "emotion category": "",
        "emotion degree": ""
    }
    emotion_dict = {
        "emotion": ""
    }
    preference_dict = {
        "preference": ""
    }
    response_dict = {
        "response": "Seeker:",
        "preference": ""
    }
    supporter_dict = {
        "similar dialog in memory": "",
        "summaried experimence": "1.xxx\n 2.xxx\n ",
        "strategy": "",
        "response": "",
    }
    question_dict = {

    }
    memory_dict = {
        "related memories": [],
        "questions": []
    }
    question_memory_dict = {
        "related memories": []
    }
    insight_dict = {
        "insights": []
    }
    experimence = {
        "experiment": ""
    }
    name,age,sex,education,hobby,nickname,occupation, appearance='','','','','','','',''
    res, emo = None, None

    initial_emo = ''

    app.run(host="0.0.0.0",port=5000)
