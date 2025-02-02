import numpy as np
import torch
from tqdm import tqdm
import os
import time
import pdb
import json
import warnings
warnings.filterwarnings("ignore")
from openai import AzureOpenAI
import httpx
from prompt import env_describe, env_situation
import re
import base64
import imageio
import random 
from utils import *

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
        
class Evaluator:
    def __init__(self, **kwargs):
        self.env = kwargs['env']
        self.device = kwargs['device']
        self.model_paths = kwargs['model_dirs']
        self.action_dim = kwargs['action_dim']
        self.env_name = kwargs['env_name']
        self.ensemble_method = kwargs['ensemble_method']
        self.epsilon = kwargs['fixed_epsilon']
        self.LLM_name = kwargs['LLM_name']
        self.sample_rate = kwargs['sample_rate']
        self.LLM_max_try = kwargs['LLM_max_try']
        self.episodes = kwargs['episodes']
        self.api_id = kwargs['api_id']
        self.frames = []
        self.if_render = True

        self.models = [DQN(self.device,self.action_dim).to(self.device) for _ in self.model_paths]
        for i, model_path in enumerate(self.model_paths):
            self.models[i].load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.models[i].eval()
        self.agent_number = len(self.models)
    
        self.boltzmann_temperature = 0.5
        self.game_dir=os.path.join('..','results',self.env_name)
        self.time_data=time.strftime('%Y-%m-%d_%H-%M', time.localtime())
        self.model_dir=f'DQN_{self.ensemble_method}_{self.time_data}'
        self.situation_num = len(env_situation[self.env_name])
        os.makedirs(os.path.join(self.game_dir,self.model_dir),exist_ok=True)
        os.makedirs(os.path.join(self.game_dir,self.model_dir,'images'),exist_ok=True)
        os.system(f'cp {__file__} ' + os.path.join(self.game_dir, self.model_dir, f'{os.path.basename(__file__)}'))
        self.models_info = []

    def set_seed(self,seed):
        torch.manual_seed(seed)
        if self.device=='cuda':
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        self.env.action_space.seed(seed)


    def gen_situation(self):
        system_prompt = 'You are a helpful reinforcement learning state classification assistant. When provided with relevant information about a RL task environment, you will categorize all states into several generalized categories. Your output format should be: {{[ENV NAME]: [ [STATE_CATEGORY 1]: [CLASSIFICATION_REASON 1], [STATE_CATEGORY 2]: [CLASSIFICATION_REASON 2], ...]}}. Ensure that your response strictly adheres to the format.'

        user_prompt = f'In the Freeway game, the primary obstacles are the moving cars on the highway. The agent needs to avoid these cars and prevent collisions. Here is an example of state classification in the Atari Freeway environment: {{"Freeway-v5": [SAFE ZONE STATE: There are no obstacles in front of the agent, and it can pass through safely, DANGER ZONE STATE: There are obstacles in front of the agent, and avoidance is needed]}}. Using the above example as a reference for classifying all states in the Atari Freeway environment, similarly classify the states in the Atari {self.env_name} environment.' + f'Below is information about the environment Atari {self.env_name}: ' + env_describe[self.env_name] + 'Please write your classification and explain the reasoning. Note: Provide only one classification method that you think is the most reasonable, with as few categories as possible. Your output format should be {{[ENV NAME]: [ [STATE_CATEGORY 1]: [CLASSIFICATION_REASON 1], [STATE_CATEGORY 2]: [CLASSIFICATION_REASON 2], ...]}}. Ensure that your output strictly adheres to the format, and DONOT conduct further reasonings.'
        ans = self.get_llm_gpt_response(system_prompt, user_prompt)
        with open('./env_situations.txt', 'a') as txt_file:
            txt_file.write(f'{self.env_name}:' + ans + "\n") 


    def get_vlm_gpt_response(self,system_prompt, user_prompt,base64_image):
        for _ in range(self.LLM_max_try):
            try:
                HTTP_CLIENT = None
                client = AzureOpenAI(
                            api_key = None,  
                            api_version = None,
                            azure_endpoint = None,
                            http_client = HTTP_CLIENT)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system", 
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": user_prompt
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                }
                            ]
                        }
                    ]
                )
                ans = response.choices[0].message.content
            except:
                ans = None
                print(f'waiting for gpt vlm response...')
                time.sleep(30)
                continue
            if len(ans)>0:
                break
            else:
                ans = None
        return ans
    
    def get_llm_gpt_response(self,system_prompt, user_prompt):
        for _ in range(self.LLM_max_try):
            try:
                HTTP_CLIENT = None
                client = AzureOpenAI(
                            api_key = None,  
                            api_version = None,
                            azure_endpoint = None,
                            http_client = HTTP_CLIENT)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system", 
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ]
                )
                ans = response.choices[0].message.content
            except:
                ans = None
                print(f'waiting for gpt vlm response...')
                time.sleep(30)
                continue
            if len(ans)>0:
                break
            else:
                ans = None
        return ans

    def if_sparse(self):
        self.if_sparse_flag = 0
        system_prompt = 'You are a helpful assistant. When provided with an image and a corresponding text prompt, you should classify the image into one of two situations and provide a bried reson for your conclusion. The output should follow this exact format:\'IfSparseReward: [IF_SPARSE_REWARD]; reason: [REASON]\', where: [IF_SPARSE_REWARD] should be replaced with 0 or 1. 0 represents non-sparse reward and 1 represents sparse reward. [REASON] should briefly explain why this classification was chosen. Ensure that your response strictly adheres to the format.'
        user_prompt = f'Below is information about the environment Atari {self.env_name}: ' + env_describe[self.env_name] + 'Please determine whether the reward in this environment is sparse. If it is sparse, output 1; if it is not, output 0. Your output format should be \'IfSparseReward: [IF_SPARSE_REWARD]; reason: [REASON]\'. The value of IF_SPARSE_REWARD is 0 or 1, where 0 represents non-sparse reward and 1 represents sparse reward. Ensure that your output strictly adheres to the format, and DONOT conduct further reasonings.'
        ans = self.get_llm_qwen_response(system_prompt,user_prompt)
        self.if_sparse_flag = int(re.findall(r'\d+', ans)[-1])
        self.if_sparse_flag = min(self.if_sparse_flag, 1)
        with open(os.path.join(self.game_dir,self.model_dir, 'if_sparse.txt'), 'a') as file:
            file.write(str(self.if_sparse_flag)+'\n\n'+ 'ANSWER=\n'+ans)
        if self.if_sparse_flag == 0:
            print('NOT Sparse reward......')
        else:
            print('Sparse reward......')

    def get_state_classify_prompt(self):
        system_prompt = 'You are a helpful assistant. When provided with an image and a corresponding text prompt, you should classify the image into one of the state categories and provide a bried reson for your conclusion. The output should follow this exact format:\'State category=[STATE_CATEGORY_ID], reason=[REASON]\', where: [STATE_CATEGORY_ID] is a number and should be replaced with the correct state classification ID. [REASON] should briefly explain why this classification was chosen. Ensure that your response strictly adheres to the format.'

        user_prompt = f'In the Atari {self.env_name} environment, many different states may occur. Below is information about the environment: ' + env_describe[self.env_name] + f" \nThe state categories faced by the agent can be divided into {self.situation_num} categories, which are listed as follows. " 
        for i in range(self.situation_num):
            user_prompt += f"{i}: {{"
            user_prompt += env_situation[self.env_name][i]
            user_prompt += '} '
        user_prompt += '\nPlease classify the input image into one of these state categories and attach brief reason for your conclusion. \nUse the output format \'State category=[STATE_CATEGORY_ID], reason=[REASON]\', where [STATE_CATEGORY_ID] is a number and should be replaced with the correct state classification ID. Ensure that your output strictly adheres to the format, and DONOT conduct further reasonings.'

        return system_prompt, user_prompt

    def find_situation_id(self,ans):
        situation_id = 0
        numbers = re.findall(r'\d+', ans)
        if numbers:
            situation_id = int(numbers[-1])
        else:
            print("No numbers found in the answer string.")
        if situation_id>self.situation_num-1:
            situation_id=self.situation_num-1
        return situation_id

    def LLM_eval_agents(self):
        def distribute_rewards(lst):
            zero_count = 0  
            for i in range(len(lst)):
                if lst[i] != 0:
                    total_positions = zero_count + 1  
                    distributed_reward = lst[i] / total_positions
                    for j in range(i - zero_count, i + 1):
                        lst[j] = distributed_reward
                    zero_count = 0
                elif lst[i] == 0:
                    zero_count += 1
            return lst
        self.if_sparse()
        if not os.path.exists(f'./model_info/{self.env_name}'):
            os.mkdir(f'./model_info/{self.env_name}')

        system_prompt, user_prompt = self.get_state_classify_prompt()

        for i in range(len(self.models)):
            model_rewards = []
            sta_infos = [] 
            for episode in tqdm(range(self.episodes)):
                if os.path.exists(os.path.join(f'./model_info/{self.env_name}',f'reward_distribution_model{i}_{episode}.json')):
                    with open(os.path.join(f'./model_info/{self.env_name}',f'reward_distribution_model{i}_{episode}.json'), 'r', encoding='utf-8') as file:
                        sta_info = json.load(file)
                    sta_infos.append(sta_info)
                    continue

                reward_history = []
                rewards_distribution_step = {}
                for j in range(self.situation_num):
                    rewards_distribution_step[j] = []

                episode_seed = random.randint(1, 5000)
                self.set_seed(episode_seed)
                state, _ = self.env.reset(seed=episode_seed)
                done = False
                total_reward = 0
                step_cnt = 0
                state = torch.tensor([state], device=self.device).float()

                while not done:
                    if step_cnt % self.sample_rate == 0 or step_cnt == 10:
                        print(f'model {i} step_cnt=', step_cnt)
                        frame = self.env.render()
                        file_path = os.path.join(self.game_dir,self.model_dir,'images', f'{episode}_frame_{step_cnt}.jpg')
                        imageio.imwrite(file_path, frame)
                        base64_image = encode_image(file_path)
                        ans = self.get_vlm_gpt_response(system_prompt, user_prompt,base64_image)
                        situation_id = self.find_situation_id(ans)

                        with open(os.path.join(self.game_dir,self.model_dir, 'llm_text_eval.txt'), 'a') as file:
                            file.write(f'\n=======step_{step_cnt}========Answer===========\n'+ans+'\n\n')
                        with open(os.path.join(self.game_dir,self.model_dir, 'situation_id_eval.txt'), 'a') as file:
                            file.write(f'step:{step_cnt}\n'+f'situation_id:{situation_id}'+'\n')

                    if np.random.random() < self.epsilon:
                        action = self.env.action_space.sample()
                    else:
                        action = np.argmax(self.models[i](state).cpu().detach().numpy())
                    next_state, reward, done, _, _ = self.env.step(action)

                    reward_history.append(reward)
                    rewards_distribution_step[situation_id].append(step_cnt)
                    step_cnt += 1
                    state = torch.tensor([next_state], device=self.device).float()
                    total_reward += reward

                    if done or step_cnt >= 1000:
                        print(f'{self.env_name} model_{i}_epi{episode} done...')
                        model_rewards.append(total_reward)
                        rewards_distribution = {} 
                        for j in range(self.situation_num):
                            rewards_distribution[j] = []
                        if self.if_sparse_flag:
                            reward_history = distribute_rewards(reward_history)
                        for j in range(self.situation_num):
                            for q in rewards_distribution_step[j]:
                                rewards_distribution[j].append(reward_history[q])
                        sta_info = {}
                        for k,v in rewards_distribution.items():
                            sta_info[k] = {}
                            sta_info[k]['mean'] = np.mean(v)
                            sta_info[k]['std'] = np.std(v)
                        self.models_info.append(sta_info)
                        with open(os.path.join(f'./model_info/{self.env_name}',f'reward_distribution_model{i}_{episode}.json'), 'w') as json_file:
                            json.dump(sta_info, json_file, indent=4)
                        sta_infos.append(sta_info)
                        break
            result = {}
            for key in sta_infos[0].keys():
                mean_avg = sum(d[key]["mean"] for d in sta_infos) / len(sta_infos)
                result[key] = mean_avg
            with open(os.path.join(f'./model_info/{self.env_name}',f'reward_distribution_model{i}_total.json'), 'w') as json_file:
                json.dump(result, json_file, indent=4)
            print(f'{self.env_name} model_{i} done...')

    
    def evaluate_LLM(self):
        self.models_info_dir = f'./model_info/{self.env_name}'
        means = [[] for j in range(self.situation_num)]
        rule_ensemble = {}
        for i in range(self.agent_number): 
            with open(os.path.join(self.models_info_dir,f'reward_distribution_model{i}_total.json'), 'r', encoding='utf-8') as file:
                reward_info = json.load(file)
            for j in range(self.situation_num):
                means[j].append(reward_info[str(j)])
        for j in range(self.situation_num):
            rule_ensemble[j] = min([i for i, v in enumerate(means[j]) if v == max(means[j])])
        
        episode_rewards = []
        system_prompt,user_prompt = self.get_state_classify_prompt()
        for episode in tqdm(range(self.episodes)):
            episode_seed = random.randint(1,5000)
            self.set_seed(episode_seed)
            state, _ = self.env.reset(seed=episode_seed)
            done = False
            total_reward = 0
            state = torch.tensor([state], device=self.device).float()
            step_cnt = 0
            situation_id = 0
            while not done:
                if self.if_render:
                    frame = self.env.render()
                    self.frames.append(frame)
                if np.random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    if step_cnt % self.sample_rate == 0:
                        print('step_cnt=', step_cnt,'inferencing......')
                        file_path = os.path.join(self.game_dir,self.model_dir,'images', f'{episode}_frame_{step_cnt}.jpg')
                        imageio.imwrite(file_path, frame)
                        base64_image = encode_image(file_path)

                        ans = self.get_vlm_gpt_response(system_prompt, user_prompt,base64_image)
                        situation_id = self.find_situation_id(ans)

                        with open(os.path.join(self.game_dir,self.model_dir, 'llm_text_evaluate.txt'), 'a') as file:
                            file.write(f'\n=======step_{step_cnt}========Answer===========\n'+ans+'\n\n')
                        with open(os.path.join(self.game_dir,self.model_dir, 'situation_id_evaluate.txt'), 'a') as file:
                            file.write(f'step:{step_cnt}\n'+f'situation_id:{situation_id}'+'\n')

                    agent_id = rule_ensemble[situation_id]
                    action = np.argmax(self.models[agent_id](state).cpu().detach().numpy())
                next_state, reward, done, _, _ = self.env.step(action)
                state = torch.tensor([next_state], device=self.device).float()
                total_reward += reward
                step_cnt += 1
                if done or step_cnt >= 1000:
                    done = True
                    episode_rewards.append(total_reward)
                    ensemble_method_temp = f'LLM_{self.sample_rate}'
                    write_csv(self.env_name,ensemble_method_temp,'1',episode_seed,total_reward,0.0)
                    imageio.mimsave(os.path.join(self.game_dir,self.model_dir,'images',f'epi{episode}_{self.ensemble_method}_{episode_seed}.gif'), self.frames, fps=30)
                    self.frames = []
                    break

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        print(f"Evaluation over {self.episodes} episodes: mean reward = {mean_reward:.2f}, std reward = {std_reward:.2f}")
        return mean_reward, std_reward