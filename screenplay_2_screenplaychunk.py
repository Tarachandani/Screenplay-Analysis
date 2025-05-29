# Script to chunk 2012_script.txt into groups of 5 lines

input_filename = '2012_script.txt'
output_filename = '2012_script_chunks.txt'
chunk_size = 15

def chunk_script(input_file, output_file, chunk_size):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(0, len(lines), chunk_size):
            chunk_number = (i // chunk_size) + 1
            f.write(f"###script_chunk {chunk_number}##\n")
            chunk = lines[i:i+chunk_size]
            for line in chunk:
                f.write(line)
            f.write("\n")  # Add extra newline between chunks for readability

if __name__ == "__main__":
    chunk_script(input_filename, output_filename, chunk_size)

