import numpy as np
import torch
import torch.nn.functional as F
import ast
import concurrent.futures
from collections import Counter
from utils.chat_utils import chat_with_gpt, chat_with_gpt4v
from agents.system_prompt import (
    Parsing_prompt,
    discriminator_prompt,
    generator_prompt
    )


class Equilibrium_Ranking():
    r""" EQUILIBRIUM - RANKING : LM R ANKING VIA EQUILIBRIUM SEARCH
    1. generate initial policies (only focus on the signalling policies)
    2. update policies using No-regret learning algorithms
    3. make decision based on the learned policies
    
    Args:
    """
    def __init__(self, text_queries, logger=None):
        
        # Hyperparameters
        self.lambda_G = 0.1
        self.lambda_D = 0.1
        self.eta_G = 0.1
        self.eta_D = 0.1
        self.iterations = 500
        self.bise = 0.05
        
        # target_hint = "Hint: the target category is " + object_category
        
        self.text_description = text_queries #+ target_hint
        message = []
        message.append({"role": "system", "content": Parsing_prompt})
        message.append({"role": "user", "content": self.text_description})
        self.response_message = chat_with_gpt(message)
        
        if logger != None:
            self.logger = logger
        
    def equilibrium_search(self, base64_image_list, candidate_id):

        self.candidate_id = candidate_id
        chat_history_for_generator_correct = []
        chat_history_for_discriminator = []
        
        chat_history_for_generator_correct.append({"role": "system", "content": generator_prompt})
        chat_history_for_discriminator.append({"role": "system", "content": discriminator_prompt})
        
        # generate initial policies 
        # Get generative probabilities for "correct"
        image_contents = []
        image_contents.append({
          "type": "text",
          "text": self.text_description,
        })
        for base64_image in base64_image_list:
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        
        chat_history_for_generator_correct.append({"role": "user", "content": image_contents})
        generative_idx, generative_probabilities_correct = self._get_generative_probabilities(chat_history_for_generator_correct)
        
        generative_probabilities_incorrect = [1 - value for value in generative_probabilities_correct]
        
        if len(generative_idx) == 1:
            return -1
        
        
        discriminative_probabilities = self._get_discriminative_probabilities(chat_history_for_discriminator, base64_image_list, generative_idx)
        
        plm_y_given_x_v = {
            'correct': [x + self.bise for x in generative_probabilities_correct],
            'incorrect': [x + self.bise for x in generative_probabilities_incorrect]
        }
        plm_v_given_x_y = discriminative_probabilities
        
        
        pi_G_1 = self._normalize_initial_policy(plm_y_given_x_v, axis=0)
        pi_D_1 = self._normalize_initial_policy(plm_v_given_x_y, axis=0)
        pi_G = pi_G_1
        pi_D = pi_D_1
        # Print initial policies
        print("Initial Generator Policy:", pi_G_1)
        print("Initial Discriminator Policy:", pi_D_1)
        
        v_values = ['correct', 'incorrect']
        y_values = list(plm_v_given_x_y.keys())
        
        # Initialize Q values
        Q_G = {v: np.zeros(len(y_values)) for v in v_values}
        Q_D = {y: np.zeros(len(v_values)) for y in y_values}
        
        # Equilibrium ranking
        for t in range(1, self.iterations + 1):
            # Update Q values
            for v in v_values:
                for i, y in enumerate(y_values):
                    Q_G[v][i] += (1 / (2 * t)) * pi_D[y][v_values.index(v)]

            for y in y_values:
                for i, v in enumerate(v_values):
                    Q_D[y][i] += (1 / (2 * t)) * pi_G[v][y_values.index(y)]

            # Update policies
            pi_G = self._update_policy(Q_G, pi_G_1, self.lambda_G, self.eta_G, t)
            pi_D = self._update_policy(Q_D, pi_D_1, self.lambda_D, self.eta_D, t)

        # Display the final policies
        print("Final Generator Policy:", pi_G)
        print("Final Discriminator Policy:", pi_D)
        
        return generative_idx[pi_G['correct'].argmax(axis=0)]
        # else:
        #     return generative_probabilities_correct.index(max(generative_probabilities_correct))
                        
    def generator_search(self, base64_image_list, candidate_id, num_samples=1):

        self.candidate_id = candidate_id
        
        chat_history_for_generator_correct = []
        
        chat_history_for_generator_correct.append({"role": "system", "content": generator_prompt})
        
        # generate initial policies 
        # Get generative probabilities for "correct"
        image_contents = []
        image_contents.append({
          "type": "text",
          "text": self.text_description,
        })
        for base64_image in base64_image_list:
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
        chat_history_for_generator_correct.append({"role": "user", "content": image_contents})
        generative_probabilities_correct = self._get_generative_probabilities(chat_history_for_generator_correct, num_samples=num_samples)
        
        max_index = generative_probabilities_correct.index(max(generative_probabilities_correct))
        return max_index

    # Normalizing initial policies
    def _normalize_initial_policy(self, plm_dict, axis=0):
        norm_dict = {}
        for key in plm_dict:
            norm_dict[key] = plm_dict[key] / np.sum(plm_dict[key], axis=axis)
        return norm_dict
    
    # Function to update policies
    def _update_policy(self, Q, pi, lambd, eta, t):
        updated_pi = {}
        for key in Q:
            exponent = (Q[key] + lambd * np.log(pi[key])) / (1 / (eta * t) + lambd)
            updated_pi[key] = np.exp(exponent) / np.sum(np.exp(exponent))
        return updated_pi

    def _get_generative_probabilities(self, chat_history, num_samples=5):
        id_list = []
        responses = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_message = {executor.submit(chat_with_gpt4v, chat_history): chat_history for _ in range(num_samples)}
            for future in concurrent.futures.as_completed(future_to_message):
                prompt = future_to_message[future]
                try:
                    response = future.result()
                    responses.append(response)
                except Exception as exc:
                    print(f'{prompt} generated an exception: {exc}')

        for i, response in enumerate(responses):
            self.logger.info(response)
            ground_json = ast.literal_eval(response)
            id_list.append(ground_json["object_id"]) 
            
        counts = Counter(id_list)
        total_count = len(id_list)
        # Calculate the probabilities
        probabilities = {number: count / total_count for number, count in counts.items()}

        probabilities_idx = []
        probabilities_value = []
        for i in (self.candidate_id ):
            if str(i) in probabilities:
                probabilities_idx.append(i)
                probabilities_value.append(probabilities[str(i)])
        print('probabilities_idx: ', probabilities_idx)
        probabilities_idx.append(-1)
        if '-1' in probabilities:
            probabilities_value.append(probabilities['-1'])
        else:
            probabilities_value.append(0)
        return probabilities_idx, probabilities_value
        
    def _get_discriminative_probabilities(self, chat_history, base64_image_list, generative_idx, num_samples=5):
        probabilities = {}
        responses = []
        gene_idx = 0
        for i, idx in enumerate(self.candidate_id):
            if idx in generative_idx:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_to_prompt = {executor.submit(chat_with_gpt4v, chat_history + self.create_payload(self.text_description, [base64_image_list[i]])): base64_image_list[i] for _ in range(num_samples)}
                    for future in concurrent.futures.as_completed(future_to_prompt):
                        prompt = future_to_prompt[future]
                        try:
                            response = future.result()
                            responses.append(response)
                        except Exception as exc:
                            print(f'{prompt} generated an exception: {exc}')
                            
                probabilities[gene_idx] = 0
                for i, response in enumerate(responses):
                    self.logger.info(response)
                    ground_json = ast.literal_eval(response)
                    discriminate = ground_json["correctness"] 
                    if discriminate == "correct":
                        probabilities[gene_idx] +=1
                    else:
                        probabilities[gene_idx] += 0
                        
                gene_idx += 1

        if gene_idx == 1:
            probabilities[-1] = num_samples - probabilities[0]
        else:
            responses = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_prompt = {executor.submit(chat_with_gpt4v, chat_history + self.create_payload(self.text_description, base64_image_list)): base64_image_list for _ in range(num_samples)}
                for future in concurrent.futures.as_completed(future_to_prompt):
                    prompt = future_to_prompt[future]
                    try:
                        response = future.result()
                        responses.append(response)
                    except Exception as exc:
                        print(f'{prompt} generated an exception: {exc}')
            probabilities[-1] = 0            
            for i, response in enumerate(responses):
                ground_json = ast.literal_eval(response)
                discriminate = ground_json["correctness"] 
                if discriminate == "incorrect":
                    probabilities[-1] +=1
                else:
                    probabilities[-1] += 0

                
        print(probabilities)            
        return {key:[value/num_samples + self.bise, 1-(value/num_samples) + self.bise] for key, value in probabilities.items()}
    
    # def _get_discriminative_probabilities(self, chat_history, base64_image_list, num_samples=5):
    #     probabilities = {}
    #     for _ in range(num_samples):
    #         responses = []
    #         with concurrent.futures.ThreadPoolExecutor() as executor:
    #                 future_to_prompt = {executor.submit(chat_with_gpt4v, self.create_payload(chat_history, self.text_description, base64_image)): base64_image for base64_image in base64_image_list}
    #                 for future in concurrent.futures.as_completed(future_to_prompt):
    #                     prompt = future_to_prompt[future]
    #                     try:
    #                         response = future.result()
    #                         responses.append(response)
    #                     except Exception as exc:
    #                         print(f'{prompt} generated an exception: {exc}')
                            
    #         incorrect_num = 0
    #         for i, response in enumerate(responses):
    #             # self.logger.info(response)
    #             ground_json = ast.literal_eval(response)
    #             discriminate = ground_json["correctness"] 
    #             if discriminate == "correct":
    #                 if i not in probabilities:
    #                     probabilities[i] = 1 
    #                 else:
    #                     probabilities[i] += 1
    #             else:
    #                 incorrect_num += 1
    #                 if i not in probabilities:
    #                     probabilities[i] = 0
    #                 else:
    #                     probabilities[i] += 0
                        
    #         if incorrect_num == len(responses): # reject all the objects
    #             if -1 not in probabilities:
    #                 probabilities[-1] = 1
    #             else:
    #                 probabilities[-1] += 1
    #         else:
    #             probabilities[-1] = 0
                
    #     print(probabilities)            
    #     return {key:[value/num_samples, 1-(value/num_samples)] for key, value in probabilities.items()}
    
    def discriminate_text(self, id):
        return "The candidate object {:d} match the user's description. Is it correct?".format(id) + '\n'



    def create_payload(self, text, base64_image_list):
        image_contents = []
        image_contents.append({
          "type": "text",
          "text": text,
        })
        for base64_image in base64_image_list:
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                    }
            })
        
        return [{"role": "user", "content": image_contents}]