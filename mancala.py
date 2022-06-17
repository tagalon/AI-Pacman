playing = True
binAmount = [4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0]

playerOne = True
messageCode = 0
while playing:

    if playerOne and messageCode == 0:
        message = "Player One's turn..."
    elif (not playerOne) and messageCode == 0:
        message = "Player Two's turn..."
    elif (not playerOne) and messageCode == -2:
        message = "Try again. Player Two's turn..."
    elif playerOne and messageCode == -2:
        message = "Try again. Player One's turn..."
    print("")
    print(message)
    print("")
    i = 0
    for element in binAmount:
        binAmount[i] = int(binAmount[i])
        if int(binAmount[i]) < 10:
            binAmount [i] = " " + str(binAmount[i])
        else:
            binAmount[i] = str(binAmount[i])
        i = i + 1
    if not playerOne:
        print("        a    b    c    d    e    f")
    print("+----+----+----+----+----+----+----+----+")
    print("|    | "+binAmount[12]+" | "+binAmount[11]+" | "+binAmount[10]+" | "+binAmount[9]+" | "+binAmount[8]+" | "+binAmount[7]+" |    |")
    print("| "+binAmount[13]+" |----+----+----+----+----+----| "+binAmount[6]+" |")
    print("|    | "+binAmount[0]+" | "+binAmount[1]+" | "+binAmount[2]+" | "+binAmount[3]+" | "+binAmount[4]+" | "+binAmount[5]+" |    |")
    print("+----+----+----+----+----+----+----+----+")
    print("")
    if (playerOne):
        print("        f    e    d    c    b    a")
    userInput = input("Enter 'q' to QUIT the game: ")
    chosenBin = 0
    if userInput == "q":
        playing = False
    elif playerOne and userInput == "a":
        chosenBin = 5
    elif playerOne and userInput == "b":
        chosenBin = 4
    elif playerOne and userInput == "c":
        chosenBin = 3
    elif playerOne and userInput == "d":
        chosenBin = 2
    elif playerOne and userInput == "e":
        chosenBin = 1
    elif playerOne and userInput == "f":
        chosenBin = 0
    elif (not playerOne) and userInput == "a":
        chosenBin = 12
    elif (not playerOne) and userInput == "b":
        chosenBin = 11
    elif (not playerOne) and userInput == "c":
        chosenBin = 10
    elif (not playerOne) and userInput == "d":
        chosenBin = 9
    elif (not playerOne) and userInput == "e":
        chosenBin = 8
    elif (not playerOne) and userInput == "f":
        chosenBin = 7
    else:
        chosenBin = -2
        messageCode = -2
    
    if int(chosenBin) >= 0:
        giveawayPile = binAmount[chosenBin]
        binAmount[chosenBin] = 0
        playerOne = not(playerOne)