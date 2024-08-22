def split_file_by_breaker(input_file, breaker):
    with open(input_file, 'r') as file:
        content = file.read()
    
    parts = content.split(breaker)
    
    for i, part in enumerate(parts):
        with open(f'output_{i+1}.out', 'w') as output_file:
            output_file.write(part)
    
    print(f"File split into {len(parts)} parts.")

input_file = 'slurm-29654716.out'
breaker = 'BREAK'
split_file_by_breaker(input_file, breaker)