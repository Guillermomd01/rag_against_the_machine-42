from transformers import AutoTokenizer, AutoModelForCausalLM

class AnswerGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_answer(self, query: str, context_chunks: list[str]) ->str:
        prompt = f"""Answer the user's question using ONLY
        the following context information. If you don't know
        the answer based on the context, say "I don't know."
        Context:
        {'\n'.join(context_chunks)}
        Question:
        {query}
        Answer:"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=200)
        input_lenght = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_lenght:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        answer_clean = answer.split("Question:")[0].strip()
        return answer_clean
        