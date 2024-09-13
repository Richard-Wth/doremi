import zstandard
import os

subset_data = "/home/wth/My_codes/doremi/data/slimpajama/subdataset/zst"
output_path = "/home/wth/My_codes/doremi/data/slimpajama/subdataset/train"

def decompress_zst_file(input_file, output_file):
    with open(input_file, "rb") as compressed_file:
        dctx = zstandard.ZstdDecompressor()
        with open(output_file, "wb") as decompressed_file:
            dctx.copy_stream(compressed_file, decompressed_file)

for file in os.listdir(subset_data):
    file_path = os.path.join(subset_data, file)
    out_file_name = file.replace(".zst", "")
    output_file = os.path.join(output_path, out_file_name)
    decompress_zst_file(input_file=file_path, output_file=output_file)
    print(f"File decompressed: {output_file}")

