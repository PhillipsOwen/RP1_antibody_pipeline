from Bio import Entrez, SeqIO

# 1️⃣ Set your email (NCBI requires this)
Entrez.email = "your_email@example.com"

# 2️⃣ Define the virus and max number of sequences to fetch
virus_name = "SARS-CoV-2"  # change this to any virus
max_sequences = 10         # increase as needed

# 3️⃣ Search NCBI Nucleotide database
search_handle = Entrez.esearch(
    db="nucleotide",
    term=f"{virus_name}[Organism]",
    retmax=max_sequences
)
search_results = Entrez.read(search_handle)
search_handle.close()
id_list = search_results["IdList"]

print(f"Found {len(id_list)} sequences for {virus_name}.")

# 4️⃣ Fetch the sequences in FASTA format
fetch_handle = Entrez.efetch(
    db="nucleotide",
    id=id_list,
    rettype="fasta",
    retmode="text"
)
sequences = fetch_handle.read()
fetch_handle.close()

# 5️⃣ Save to a file
output_file = f"{virus_name.replace(' ', '_')}_sequences.fasta"
with open(output_file, "w") as f:
    f.write(sequences)

print(f"Sequences saved to {output_file}")