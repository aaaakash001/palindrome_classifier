import csv

def generate_binary_strings(length):
    binary_strings = []
    for i in range(2**length):
        binary_string = bin(i)[2:].zfill(length)
        binary_strings.append(binary_string)
    return binary_strings

def is_palindrome(string):
    return string == string[::-1]

length = 10
binary_strings = generate_binary_strings(length)


csv_file_path = "binary_strings_palindrome_check.csv"

with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Write header
    csv_writer.writerow(["Binary String", "Palindrome Check"])
    
    # Write data
    for binary_string in binary_strings:
        palindrome_check = "Palindrome" if is_palindrome(binary_string) else "Not Palindrome"
        csv_writer.writerow([binary_string, palindrome_check])

print(f"CSV file saved at: {csv_file_path}")
