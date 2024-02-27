import json
import os
import openai
import argparse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

# Define model endpoints
MODEL_ENDPOINTS = {
    'openai': ['gpt-4', 'gpt-4-1106-preview', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo'],
    'google': ['gemini-pro']
}


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_ans(model, module, prompt, parser):
    if model in MODEL_ENDPOINTS['openai']:
        llm = ChatOpenAI(model_name=model, verbose=True)
    elif model in MODEL_ENDPOINTS['google']:
        llm = ChatGoogleGenerativeAI(model=model, verbose=True)

    ans = llm.invoke(input=prompt)
    ans = ans.content

    if module == 'translate':
        ans_dict = parser.parse(ans)
        ans_mt = ans_dict['Target']
        # print(f"Translate: {ans_mt}")
        return ans_mt

    elif module == 'estimate':
        ans_dict = parser.parse(ans)
        all_no_error = all(value == "no-error" or value == '' or value == "null" or value == None for value in ans_dict.values())
        if all_no_error:
            # print("No need for correction")
            nc = 0
        else:
            nc = 1
        # print(f"Estimate: {ans}")
        return ans, nc

    elif module == 'refine':
        ans_dict = parser.parse(ans)
        ans_mt = ans_dict['Final Target']
        # print(f"Refine: {ans_mt}")
        return ans_mt

class TER:
    def __init__(self, lang_pair, model, module, strategy, prompt_path='./prompts/'):
        self.lang_pair = lang_pair
        self.model = model
        self.module = module
        self.strategy = strategy
        with open(os.path.join(prompt_path, f'{self.module}/{self.strategy}.txt'), 'r', encoding='utf-8') as f:
            raw_template = f.read()
        self.template = self.form_template(raw_template)

    def form_template(self, raw_template):
        temp = PromptTemplate.from_template(template=raw_template)
        return temp

    def load_examples(self):
        try:
            if self.strategy == 'few-shot':
                qr_fewshot_path = f"prompts/data-shots/mt/shots.{self.lang_pair}.json"
                with open(qr_fewshot_path, 'r', encoding='utf-8') as json_file:
                    few = json.load(json_file)

                cases_srcs = [few[i]['source'] for i in range(len(few))]
                cases_tgts = [few[i]['target'] for i in range(len(few))]
                origin_template = '''Source: {src_top} Target: {tgt_ans}'''

                cases_formatted = '\n'.join([
                    origin_template.format(src_top=cases_srcs[i], tgt_ans=cases_tgts[i]) for i in range(len(cases_srcs))])

            elif self.strategy == 'zero-shot':
                cases_formatted = ''
            else:
                cases_formatted = ''
        except: 
            cases_formatted = ''

        return cases_formatted

    def set_parser(self):
        if self.module == 'translate':
            ans_schema = ResponseSchema(name="Target", description="The final translation. Please use escape characters for the quotation marks in the sentence.")
            json_parser = StructuredOutputParser.from_response_schemas([ans_schema])
            json_output_instructions = json_parser.get_format_instructions()
        elif self.module == 'estimate':
            cr = ResponseSchema(name="critical", description="critical errors")
            mj = ResponseSchema(name="major", description="major errors")
            mn = ResponseSchema(name="minor", description="minor errors")

            ans_schema = [cr, mj, mn]
            json_parser = StructuredOutputParser.from_response_schemas(ans_schema)
            json_output_instructions = json_parser.get_format_instructions()
        elif self.module == 'refine':
            ans_schema = ResponseSchema(name="Final Target", description="The final translation. Please use escape characters for the quotation marks in the sentence.")
            json_parser = StructuredOutputParser.from_response_schemas([ans_schema])
            json_output_instructions = json_parser.get_format_instructions()

        return json_parser, json_output_instructions

    def fill_prompt(self, src_lan, tgt_lan, src, json_output_instructions, examples=None, hyp=None, mqm_info=None):
        if self.module == 'translate':
            write_prompt = self.template
            prompt = write_prompt.format(src_lan=src_lan, tgt_lan=tgt_lan, examples=examples, origin=src.strip(), format_instructions=json_output_instructions)
        elif self.module == 'estimate':
            write_prompt = self.template
            prompt = write_prompt.format(src_lan=src_lan, tgt_lan=tgt_lan, origin=src.strip(), trans=hyp, format_instructions=json_output_instructions)
        elif self.module == 'refine':
            write_prompt = self.template
            prompt = write_prompt.format(src_lan=src_lan, tgt_lan=tgt_lan, examples=examples, raw_src=src.strip(), raw_mt=hyp, sent_mqm=mqm_info, format_instructions=json_output_instructions)
        return prompt

