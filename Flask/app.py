from flask import Flask, render_template, request, jsonify, redirect, url_for, session # For Web GUI
import pickle # For object serialization

import prediction
import collaborativeFiltering

app = Flask(__name__) # Set up app
#recommendationModel = pickle.load(open('recommendationModel.pkl', 'rb')) # Load serialized model

app.secret_key = 'optimizePrime09314873'

recommendedMovies = []

userID = None
userIsLoggedIn = False

# Set main default route to render index
@app.route('/', methods = ['POST', 'GET'])
def index():
	'''
	Shows the form for inserting input movie
	'''
	if request.method == 'POST':
		# Obtain movie name, from form
		movie_name = request.form['movie']

		# Obtain prediction movies
		try:
			x = prediction.get_recommendations(movie_name).to_list()

			# Redirect
			return render_template('index.html', recommendedMovies = enumerate(x), numberOfMovies = len(x), showFirstTable = True)

		except: # Movie title was not found
			x = []
			# Redirect
			return render_template('index.html', recommendedMovies = enumerate(x), numberOfMovies = len(x), MovienotFoundError = True, inputMovie = movie_name)

	else:
		recommendedMovies = []
		# If we use a database, then at this line it would query the elements
		return render_template('index.html', recommendedMovies = enumerate(recommendedMovies), numberOfMovies = len(recommendedMovies))

# Set route for movies I like request
@app.route('/moviesILike', methods = ['POST'])
def moviesILike():
	
	# Obtain user ID, from form
	userIDInput = request.form['userIDCollaborativeFiltering']

	# validate inputs
	if validInputsLoginForm(userIDInput) != 'Valid inputs':
		return render_template('login.html', loginError = True, errorMessage = validInputsLoginForm(userIDInput))
	else: # cast to integer
		userIDInput = int(userIDInput)


	# Obtain recommended movies
	try:
		
		x = collaborativeFiltering.get_recommendations_per_user(userIDInput)

		# Redirect
		return render_template('index.html', recommendedMovies = enumerate(x), numberOfMovies = len(x), showSecondTable = True)

	except: # User was not found
		x = []
		# Redirect
		return render_template('index.html', recommendedMovies = enumerate(x), numberOfMovies = len(x))

# API mode using JSON
@app.route('/recommend_api', methods = ['POST'])
def recommend_api():
	'''
	Allows API requests
	'''

	parameters = request.get_json() # get_json()

	movieName = parameters['movie']

	recommendedResults = prediction.get_recommendations(movieName).to_list()
	
	return jsonify({'result' : 'Success!', 'movie received' : movieName, 'results': recommendedResults}) # recommendedResults # jsonify

# Login route
@app.route('/login', methods = ['POST', 'GET'])
def login():
	if request.method == 'GET': # Show login site
		return render_template('login.html')
	else:
		userID = request.form['userID']

		# Validate given input
		if validInputsLoginForm(userID) != 'Valid inputs':
			return render_template('login.html', loginError = True, errorMessage = validInputsLoginForm(userID))
		else: # cast to integer
			userID = int(userID)

		# Check if user exists
		if validateUserID(userID): # user exists
			session['userID'] = userID
			userIsLoggedIn = True
			return render_template('login.html', loginSuccesful = True, userIsLoggedIn = True)
			
		# User does not exist
		return render_template('login.html', loginError = True, errorMessage = "User does not exit")

# Function to validate the input of the login form
def validInputsLoginForm(userID):
	# empty
	if userID == '':
		return 'Please insert an ID value'

	# try casting input to integer, else, show error that numerical values are only accepted
	try:
		userID = int(userID)
	except:
		return 'Please insert only numerical values'

	return 'Valid inputs'

# Function to validate login of a valid user id
def validateUserID(userID):
	# Retrieve list of valid IDs
	listUserIDs = collaborativeFiltering.get_user_IDs()

	if userID in listUserIDs:
		return True
	else:
		return False


# Logout route
@app.route('/logout', methods = ['POST', 'GET'])
def logout():
	session.pop('userID', None)
	return redirect(url_for('login'))

if __name__ == '__main__':
	app.run(debug = True)