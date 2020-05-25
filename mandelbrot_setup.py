def userInput():
    while True:
        try:
            print('Zoom level (0 - 5) 0 is original 5 i zoomed max in: ')
            user_input = int(input())       
        except ValueError:
            print("Not an integer! Try again.")
            continue
    
        try:
            print('What would you like to call the file? Leave empty for "default" \nIf filename already exists then it will be overwritten: ')
            file_name = input()
            if len(file_name) < 1:
                file_name = "default"    

        except ValueError:
            print("Error with the name, Try again.")
            continue

        else:
            return user_input, file_name
            break 