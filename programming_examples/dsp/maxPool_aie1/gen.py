# Open the file in write mode
with open("input.txt", "w") as file:
    # Loop through the range from 0 to 1151
    for i in range(1152):
        # Write the number, cycling through 0 to 255 using modulo
        file.write(f"{i % 9}\n")