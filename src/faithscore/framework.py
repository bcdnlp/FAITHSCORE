
import openai
import time
from tqdm import tqdm
import argparse
import os
import re
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from modelscope.preprocessors.multi_modal import OfaPreprocessor
from faithscore.llava15 import LLaVA
from faithscore.llama_pre import load_llama, stage1_llama, stage2_llama
from faithscore.utils import llava15, ofa
import nltk


path = os.path.dirname(__file__)
cur_path = os.path.dirname(path)
cur_path = os.path.join(cur_path, "faithscore")
class FaithScore():
    def __init__(self, vem_type, api_key=None, llava_path=None, tokenzier_path=None, use_llama=False, llama_path=None):
        openai.api_key = api_key
        max_seq_len = 500
        max_batch_size = 1
        self.use_llama = use_llama

        # self.vem_path = model_path
        self.model_type = vem_type ### [ofa_ve, ofa, mplug, blip2, llava]
        model_list = ["ofa_ve", "ofa", "mplug", "blip2", "llava"]
        if vem_type not in model_list:
            print(f"Error: the model type {vem_type} not in {str(model_list)}")
            exit()
        self.llava_path = llava_path
        
        if use_llama:
            if llama_path:
                self.llama, self.tokenizer = load_llama(llama_path)
            else:
                print(f"Error: please input the model path for llama")
                exit()

    def call_openai(self, pts):
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user",
                         "content": pts},
                    ],
                    temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                )
                return response['choices'][0]['message']['content']
            except Exception as e:
                print(e)
                print("Continue......")
                time.sleep(10)

    def stage1(self, answers):
        with open(os.path.join(cur_path, "prompts/prompt_label_des_ana.txt"), "r") as f:
            prompt_label_des_ana = f.read() + "\n\n"
        des_ana = []
        print("Stage 1: Sub-sentence Identification")
        if self.use_llama:
            des_ana = stage1_llama(self.llama, self.tokenizer, [a.replace("\n", " ") for a in answers])
        else:
            for id in tqdm(range(len(answers))):
                if not self.use_llama:
                    pts = prompt_label_des_ana + answers[id].replace("\n", " ") + "\n" + "Labeled text: "
                    des_ana.append(self.call_openai(pts).replace("\n", ""))
            # else:
            #     pts = stage1_llama(self.llama, self.tokenizer, answers[id].replace("\n", " "))
            #     # print(pts)
            #     des_ana.append(pts)
            # exit()
        return des_ana

    def stage2(self, labeld_sub_sen, ):
        all_texts = []
        lens = [len(subs) for subs in labeld_sub_sen]
        labeld_sub_sen = [s for subs in labeld_sub_sen for s in subs]
        for ss in labeld_sub_sen:
            desc = ""
            pos_des = [substr.start() for substr in re.finditer("[D]", ss)]
            pos_ana = [substr.start() for substr in re.finditer("[A]", ss)]
            pos_seg = pos_des + pos_ana
            pos_seg.sort()
            for i in range(len(pos_seg)):
                if pos_seg[i] in pos_des:
                    if i == 0:
                        desc += ss[:pos_seg[i] - 1]
                    else:
                        desc += ss[pos_seg[i - 1] + 3:pos_seg[i] - 1]
            all_texts.append(desc.replace("\n", " "))

        with open(os.path.join(cur_path, "prompts/prompt_de_atomic.txt"), 'r') as f:
            prompt_de_atomic = f.read()
        Entities = []
        Relations = []
        Colors = []
        Counting = []
        Others = []

        results = []
        nons = "Entities:\nRelations:\nColors:\nCounting:\nOther attributes:"
        print("Stage 2: Atomic Fact Generation")
        inputs = []
        for ans in tqdm(all_texts):
            ans = ans.replace("\n", " ")
            pts = prompt_de_atomic + "\nAnswer: " + ans
            inputs.append(pts)
        if self.use_llama:
            response = stage2_llama(self.llama, self.tokenizer, inputs)
        else:
            response = self.call_openai(inputs) 
        for idx, r in enumerate(response):
            if all_texts[idx] == "" or "Entities" not in r:
                response[idx] = nons

        results = response 

        for i,facts in enumerate(results):
            lines = facts.split("\n")
            entity_seen = False
            for line in lines:
                if line == "":
                    continue
                if line[:9] == "Entities:":
                    if entity_seen:
                        break
                    entity_seen=True
                    entity = line.strip().replace("Entities: ", "").split(". ")
                    if line.strip() == "Entities:":
                        entity = []
                    Entities.append(entity)
                elif line[:10] == "Relations:":
                    # print(line.strip().replace("Relations: ","").replace("],","]],").split("], "))
                    relation = line.strip().replace("Relations: ", "").split(". ")
                    if line.strip() == "Relations:":
                        relation = []
                    Relations.append(relation)
                elif line[:7] == "Colors:":
                    color = line.strip().replace("Colors: ", "").split(". ")
                    if line.strip() == "Colors:":
                        color = []
                    Colors.append(color)
                elif line[:9] == "Counting:":
                    count = line.strip().replace("Counting: ", "").split(". ")
                    if line.strip() == "Counting:":
                        count = []
                    Counting.append(count)
                elif line[:17] == "Other attributes:":
                    other = line.strip().replace("Other attributes: ", "").split(". ")
                    if line.strip() == "Other attributes:":
                        other = []
                    Others.append(other)
            for l in [Entities, Relations, Colors, Counting, Others]:
                if len(l) < i+1:
                    for i in range(i-len(l)+1):       
                        l.append([])
             
        unflattened_Entities = []
        unflattened_Relations = []
        unflattened_Colors = []
        unflattened_Counting = []
        unflattened_Others = []
        index = 0
        for length in lens:
            unflattened_Entities.append(Entities[index:index+length])
            unflattened_Relations.append(Relations[index:index+length])
            unflattened_Colors.append(Colors[index:index+length])
            unflattened_Counting.append(Counting[index:index+length])
            unflattened_Others.append(Others[index:index+length])
            index += length
        Entities = unflattened_Entities
        Relations = unflattened_Relations
        Colors = unflattened_Colors
        Counting = unflattened_Counting
        Others = unflattened_Others
        hallucinations = [Entities[i] + Relations[i] + Colors[i] + Counting[i] + Others[i] for i in range(len(Entities))]
        return hallucinations, Entities, Relations, Colors, Counting, Others

    def stage3(self, atomic_facts, images, img_path=None):
        # ofa_pipe = pipeline(Tasks.visual_entailment, model='damo/ofa_visual-entailment_snli-ve_large_en')
        # model = pipeline(Tasks.visual_entailment, model=self.vem_path)
        print("Stage 3: Verification")
        if self.model_type == "ofa_ve":
            model = pipeline(Tasks.visual_entailment, model='damo/ofa_visual-entailment_snli-ve_large_en')

        if self.model_type == "ofa":
            preprocessor = OfaPreprocessor(model_dir="damo/ofa_visual-question-answering_pretrain_large_en")
            model = pipeline(
                Tasks.visual_question_answering,
                model="damo/ofa_visual-question-answering_pretrain_large_en",
                model_revision='v1.0.1',
                preprocessor=preprocessor, 
                batch_size=8)

        if self.model_type == "llava":
            if not self.llava_path:
                print("Please input path for LLaVA model.")
                exit()
            model = LLaVA()
        # if self.model_type == "mplug":
        #     output = mplug(image, prompt, model)
        # if self.model_type == "blip2":
        #     output = blip_2(image, prompt, model, vis_processors_blip_2)
        
        fact_scores = []
        atomic_facts = [[f for f in sublist if f != []] for sublist in atomic_facts]
        lengths = [len([s for s in sublist if s != []]) for sublist in atomic_facts]
        lengths_2 = [[len(s) for s in sublist if s != []] for sublist in atomic_facts]
        flatten_attomic_facts = [item for sublist in atomic_facts for item in sublist if item != []]
        flatten_attomic_facts = [item for sublist in flatten_attomic_facts for item in sublist]
        images = [[images[i]]*sum(lengths_2[i]) for i in range(len(images))]
        flattened_images = [item for sublist in images for item in sublist]
        BS = 16
        for idx in tqdm(range(0,len(flatten_attomic_facts), BS)):
            facts = flatten_attomic_facts[idx:idx+BS]
            if img_path:
                image = [os.path.join(img_path, flattened_images[j]) for j in range(idx, min(idx+BS, len(flattened_images)))]
            else:
                image = [flattened_images[j] for j in range(idx, min(idx+BS, len(flattened_images)))]
                
            prompts = []
            for element in facts:
                # input = {'image': image, 'text': element}
                prompts.append('Statement: ' + element + ' Is this statement is right according to the image? Please answer yes or no.')
            if self.model_type == "ofa_ve":
                output = ofa(True, model, prompts, image)
            if self.model_type == "ofa":
                output = ofa(False, model, prompts, image)
            if self.model_type == "llava":
                output = llava15(image, prompts, model)
                # print(output)
                # if self.model_type == "mplug":
                #     output = mplug(image, prompt, model)
                # if self.model_type == "blip2":
                #     output = blip_2(image, prompt, model, vis_processors_blip_2)
            for out in output:
                if "yes" in out.lower():
                    fact_scores.append(1)
                else:
                    fact_scores.append(0)
                # fact_scores.append(fact_score)
                # results[id] = sum(fact_score)/len(fact_score) if len(fact_score) > 0 else 0

            # # output = ofa_pipe(input)[0]
            # if "yes" in output.lower():
            #     fact_score.append(1)
            # else:
            #     fact_score.append(0)

            #     # if output.lower() == "yes" or output== "Yes":
            #     #     fact_score.append(1)
            #     # else:
            #     #     fact_score.append(0)
            # fact_scores.append(fact_score)
            # results[id] = sum(fact_score)/len(fact_score) if len(fact_score) > 0 else 0
                # result.append(output[OutputKeys.LABELS])
            # results.append({"image": images_id[id], "facts": elements, "result": str(result)})
            # checking_results.append(result)
        unflattented_fact_scores = []
        results = {}
        index = 0
        for i,length in enumerate(lengths_2):
            unflattented_fact_scores.append(fact_scores[index:index+sum(length)])
            results[i] = sum(unflattented_fact_scores[i])/sum(length) if sum(length) > 0 else 0

            index += sum(length)
            
        # unflattented_fact_scores = []
        # index = 0
        # for i, length in enumerate(lengths):
        #     unflattented_fact_scores.append(unflattented_fact_scores_2[index:index+length])
        #     index += length
            instance_score = [sum([iii for iii in ii]) / len([iii for iii in ii]) if len([iii for iii in ii]) > 0 else 0 for ii in unflattented_fact_scores]

        # instance_score = [sum([iiii for iii in ii for iiii in iii]) / len([iiii for iii in ii for iiii in iii]) if len([iiii for iii in ii for iiii in iii]) > 0 else 0 for ii in unflattented_fact_scores]
        # print("Overall score: ", sum(instance_score) / len(instance_score))

        return sum(instance_score) / len(instance_score), unflattented_fact_scores, results
    '''
    answers: a list of strings, each element in this list is an answer
    '''

    def faithscore(self, answers, images):
        ## Stage 1: Sub-setence Identification
        labeld_sub_sen = self.stage1(answers)
        ### Stage 2: Atomic Fact Generation
        atomic_facts, Entities, Relations, Colors, Counting, Others = self.stage2(labeld_sub_sen)
        ### Stage 3: Verification
        # print(atomic_facts)
        score, fact_scores, results = self.stage3(atomic_facts, images)
        sentence_score, results_sentence = self.sentence_faithscore(Entities, Relations, Colors, Counting, Others, self.labeled_sub(labeld_sub_sen), fact_scores)
        return score, sentence_score, results, results_sentence

    def sentence_faithscore(self, Entities, Relations, Colors, Counting, Others, all_texts, fact_scores):
        Entities_recog = []
        for ents in Entities:
            entities = []
            ents = [ee for e in ents for ee in e ]
            for e in ents:
                ent4sen = []
                sentence = nltk.sent_tokenize(e)
                tags = nltk.pos_tag(nltk.word_tokenize(sentence[0]))
                for tag in tags:
                    if tag[1] in ['NN', 'NNS', 'JJ', 'NNP', 'VBG', 'JJR', 'NNPS', 'RB', 'DT']:
                        # print(tag)
                        ent4sen.append(tag[0])
                    else:
                        ent4sen.append("")
                    # tags.append(chunk.label())

                if len(ent4sen) < 1:
                    print(tags)
                    ent4sen.append("")
                    # exit()
                
                entities.append(ent4sen[-1])


            if len(entities) != len(ents):
                print("error")
                exit()
            Entities_recog.append(entities)

        entity_scores = []
        relation_scores = []
        color_scores = []
        count_scores = []
        other_scores = []
        
        clean_empty = lambda x: [ii for i in x for ii in i if i]
        Entities = [clean_empty(e) for e in Entities]
        Relations = [clean_empty(r) for r in Relations]
        Colors = [clean_empty(c) for c in Colors]
        Counting = [clean_empty(c) for c in Counting]
        Others = [clean_empty(o) for o in Others]
        

        for i in range(len(fact_scores)):
            entity_scores.append(fact_scores[i][:len(Entities[i])])
            relation_scores.append(fact_scores[i][len(Entities[i]): len(Entities[i]) + len(Relations[i])])
            color_scores.append(fact_scores[i][
                                len(Entities[i]) + len(Relations[i]):  len(Entities[i]) + len(Relations[i]) + len(
                                    Colors[i])])
            count_scores.append(fact_scores[i][
                                len(Entities[i]) + len(Relations[i]) + len(Colors[i]):  len(Entities[i]) + len(
                                    Relations[i]) + len(Colors[i]) + len(Counting[i])])
            other_scores.append(
                fact_scores[i][len(Entities[i]) + len(Relations[i]) + len(Colors[i]) + len(Counting[i]):])

        sentence_scores = []
        results = {}
        
        for id1, ins in enumerate(all_texts):
            sentence_score = []
            for id2, sub_sen in enumerate(all_texts[id1]):
                flag = True
                for id3, ee in enumerate(Entities_recog[id1]):
                    if ee in sub_sen and entity_scores[id1][id3] != 1:
                        flag = False
                    for id4, rel in enumerate(relation_scores[id1]):
                        if ee in sub_sen and ee in Relations[id1][id4] and rel != 1:
                            flag = False
                    for id4, rel in enumerate(color_scores[id1]):
                        if ee in sub_sen and ee in Colors[id1][id4] and rel != 1:
                            flag = False
                    for id4, rel in enumerate(count_scores[id1]):
                        if ee in sub_sen and ee in Counting[id1][id4] and rel != 1:
                            flag = False
                    for id4, rel in enumerate(other_scores[id1]):
                        if ee in sub_sen and ee in Others[id1][id4] and rel != 1:
                            flag = False

                sentence_score.append(flag)
            sentence_scores.append(sentence_score)
            results[id1] = sum(sentence_score)/len(sentence_score) if len(sentence_score) > 1 else sentence_score


        score4sen = [sum(ss)/len(ss) if len(ss) > 0 else 1 for ss in sentence_scores]
        sentence_level_score = score4sen
        # print(score4sen)
        # print(sum(score4sen)/len(score4sen))
        return sum(score4sen)/len(score4sen), results

    def labeled_sub(self, des_ana):
        all_texts = []
        for ss in des_ana:
            desc = []
            for sss in ss:
                pos_des = [substr.start() for substr in re.finditer("[D]", sss)]
                pos_ana = [substr.start() for substr in re.finditer("[A]", sss)]
                pos_seg = pos_des + pos_ana
                pos_seg.sort()
                for i in range(len(pos_seg)):
                    if pos_seg[i] in pos_des:
                        if i == 0:
                            desc.append(sss[:pos_seg[i] - 1])
                        else:
                            desc.append(sss[pos_seg[i - 1] + 3:pos_seg[i] - 1])
            all_texts.append(desc)
        return all_texts