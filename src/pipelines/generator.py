from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class AnswerGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("\n[Info] Cargando modelo con Accelerate (device_map='auto')...")
        
        # Accelerate se encarga de todo: busca tu GPU o CPU y lo optimiza.
        # torch_dtype=torch.float16 reduce el consumo de RAM a la mitad y duplica la velocidad.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )

    # Desactivamos el motor de entrenamiento de PyTorch
    @torch.inference_mode() 
    def generate_answer(self, query: str, context_chunks: list[str]) -> str:
        # Mantenemos un límite sano para no desbordar la memoria (VRAM/RAM)
        context_str = '\n'.join(context_chunks)
        max_context_length = 3500 
        if len(context_str) > max_context_length:
            context_str = context_str[:max_context_length] + "\n...[Context truncated]"

        prompt = f"""Answer the user's question using ONLY the following context information. If you don't know the answer based on the context, say "I don't know."
Context:
{context_str}

Question:
{query}
Answer:"""
        
        # .to(self.model.device) es vital aquí para enviar el texto al mismo hardware que eligió Accelerate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=100, 
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            use_cache=True 
        )
        
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        answer_clean = answer.split("Question:")[0].strip()
        answer_clean = answer_clean.split("Context:")[0].strip()
        
        return answer_clean