
r = open("adults.csv", "r")
r = ''.join([i for i in r])
r = r.replace("?", "Unemployed")

w = open("adults-clean.csv", "w")
w.writelines(r)
w.close()

