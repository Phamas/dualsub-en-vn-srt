import os
import argparse
try:
    from transformers import MarianMTModel, MarianTokenizer
    import torch
except ImportError:
    print("Please install required libraries by running:")
    print("pip install transformers torch sentencepiece")
    exit(1)

def translate_srt_local(input_file, batch_size=16):
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    # Automatically generate output filename
    base_name = os.path.basename(input_file)
    output_file = f"en-vn-{base_name}"

    print(f"Loading local translation model (this will download ~300MB on the first run)...")
    model_name = "Helsinki-NLP/opus-mt-en-vi"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    # Use GPU hardware acceleration if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded successfully! Executing on: {device.upper()}")

    with open(input_file, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    blocks = content.strip().split('\n\n')
    print(f"Total subtitle blocks to translate: {len(blocks)}")
    
    parsed_blocks = []
    texts_to_translate = []
    
    for block in blocks:
        lines = block.split('\n')
        if len(lines) >= 3:
            index = lines[0]
            timecode = lines[1]
            original_text = '\n'.join(lines[2:])
            
            parsed_blocks.append({
                'index': index,
                'timecode': timecode,
                'original': original_text
            })
            # Replace newlines for smoother contextual translation
            texts_to_translate.append(original_text.replace('\n', ' '))

    translated_texts = []
    
    # Process in batches for memory efficiency
    total_batches = (len(texts_to_translate) - 1) // batch_size + 1
    
    for i in range(0, len(texts_to_translate), batch_size):
        batch_texts = texts_to_translate[i:i+batch_size]
        print(f"Translating batch {i//batch_size + 1}/{total_batches}...")
        
        # Tokenize and translate
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            translated_tokens = model.generate(**inputs)
        
        # Decode translations
        batch_translations = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        translated_texts.extend(batch_translations)

    # Reconstruct SRT
    final_srt_blocks = []
    for parsed, vi_text in zip(parsed_blocks, translated_texts):
        original = parsed['original']
        combined_text = f"{original}\n<i>{vi_text}</i>"
        new_block = f"{parsed['index']}\n{parsed['timecode']}\n{combined_text}"
        final_srt_blocks.append(new_block)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(final_srt_blocks) + '\n')
        
    print(f"Translation complete! File saved as: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate English SRT to dual-language English-Vietnamese SRT using local AI models.")
    parser.add_argument("input_file", help="Path to the input SRT file")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for processing (default: 16)")
    
    args = parser.parse_args()
    
    translate_srt_local(args.input_file, args.batch)
