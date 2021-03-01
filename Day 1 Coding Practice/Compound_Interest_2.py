#Formula to calculate annual compound interest:
#A = P(1 + R/100)^t

#A is final amount
#P is principle amount
#R is the rate
#T is the time span


def compound_interest(principle, rate, time):

    #Calculates compound interest
    Amount = principle * (pow((1 + rate / 100), time))
    CI = Amount
    print("Compound interest is", CI)

#Driver Code
compound_interest(14000, 10, 5)

print(compound_interest)