from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


class AnswerGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B"):
        """Initializes the AnswerGenerator by loading the
        specified model and tokenizer, and setting up the
        device for inference."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Detect the best available device (GPU/MPS/CPU)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            print("[Warning] No se detectó GPU/MPS."
                  "Se usará la CPU.")

        print(f"\n[Info] Loading model on device:"
              f" {str(self.device).upper()}...")

        # Loading the model with the detected device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to(self.device)  # type: ignore

    @torch.inference_mode()
    def generate_answer(self, query: str, context_chunks: list[str]) -> str:
        context_str = '\n'.join(context_chunks)
        max_context_length = 3500
        if len(context_str) > max_context_length:
            context_str = context_str[
                :max_context_length] + "\n...[Context truncated]"

        prompt = f"""Answer the user's question using ONLY
        the following context information. If you don't know
        the answer based on the context, say "I don't know."
        Context: {context_str}
        Question: {query} Answer:"""

        # tensorize the prompt and move it to the same device as the model
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            use_cache=True
        )

        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        answer = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True)
        answer_str = str(answer)
        answer_clean = answer_str.split("Question:")[0].strip()
        answer_clean = answer_clean.split("Context:")[0].strip()

        return answer_clean
