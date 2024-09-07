import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
    model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Ensure pad_token_id is set to eos_token_id if pad_token_id is None
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer, model, device

# Generate code based on prompt
def generate_code(prompt, tokenizer, model, device, max_length=128, temperature=0.7, top_p=0.95):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None).to(device)
    
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_code[len(prompt):]

def main():
    st.title("Code Generation with Salesforce CodeGen")
    
    # Load model and tokenizer
    tokenizer, model, device = load_model()
    
    # Input prompt from the user
    prompt = st.text_area("Enter your prompt here:", height=200)
    
    if st.button("Generate Code"):
        if prompt.strip() == "":
            st.error("Please enter a prompt.")
        else:
            st.write("Generating code...")
            generated_code = generate_code(prompt, tokenizer, model, device)
            st.subheader("Generated Code:")
            st.code(generated_code, language='python')

if __name__ == "__main__":
    main()
