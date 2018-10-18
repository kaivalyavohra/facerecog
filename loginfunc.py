from faces import detect

name=input("Welcome to this program. Please enter your first name if you would like to log in: ").lower()
if detect().lower()!=name:
	print("Unsuccesful attempt.")
else:
	print("Succesfully logged in.")