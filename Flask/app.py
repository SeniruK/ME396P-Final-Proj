
# This app.py file contains the logic for the GUI Web interface using Flask framework

from flask import Flask, render_template, request, jsonify, redirect, url_for, session # For Web GUI

# Import backend files that contain the logic for recommending movies to the user
import prediction

app = Flask(__name__) # Set up app

app.secret_key = 'optimizePrime09314873' # A secret key is required in order to use sessions in Flask

# HELPER FUNCTIONS

def validInputsLoginForm(userID):
	'''	
		Function to validate the input of the login form.
		Given an input ID from the user, returns a String message (if it is an error or 'Valid inputs')
	'''

	if userID == '': # The user introduced an empty ID
		return 'Please insert an ID in order to login'

	# Else, try casting input to integer, if that isn't possible, show error that numerical values are only accepted
	try:
		userID = int(userID)
	except: # exception while casting (input is not a number)
		return 'Please insert only numerical values'

	return 'Valid inputs' # OK


def validateUserID(userID):
	'''
		Function to validate if the given ID exists within the list of users.
		Returns a bool in case the user exists or not.
	'''

	# Retrieve list of valid IDs
	listUserIDs = prediction.get_user_IDs()

	if userID in listUserIDs:
		return True
	else:
		return False

# ROUTES


@app.route('/', methods = ['GET'])
def indexGET():
	'''
		Set main default route to render index.html (AKA Home Screen) (GET)
		Shows the login site if the user isn't logged in yet, else it shows 'Find movies based on my likes' button
	'''

	try: # try to check if user is logged in (at the session)

		userID = session['userID'] # if user is logged in show form...

		return render_template('index.html') # index html file contains a form for finding movies based on the current user

	except: # Else, if user isn't logged in, show him the login site

		return render_template('login.html', 
			loginError = True, 
			errorMessage = validInputsLoginForm("")) # Show an error, in order for the user to know it has to log in


@app.route('/', methods = ['POST'])
def indexPOST():
	'''
		Set main default route to render index (POST).
		It is called when the user clicks on the 'Find movies based on my likes' button
	'''
	
	# Obtain user ID, from form
	userIDInput = request.form['userIDCollaborativeFiltering']

	# validate inputs
	if validInputsLoginForm(userIDInput) != 'Valid inputs': # invalid inputs, show error
		return render_template('login.html', 
			loginError = True, 
			errorMessage = validInputsLoginForm(userIDInput))
	else: # cast to integer
		userIDInput = int(userIDInput)

	# Obtain recommended movies
	try:
		results = prediction.get_recommendations_per_user(userIDInput)

		# Success! Show list with results (enumerate to show index on the table)
		return render_template('index.html', 
			recommendedMovies = enumerate(results), 
			numberOfMovies = len(results), 
			showSecondTable = True)

	except: # User was not found, show error
		results = []
		# Redirect
		return render_template('index.html', 
			recommendedMovies = enumerate(results), 
			numberOfMovies = len(results))



@app.route('/byOverview', methods = ['GET'])
def byOverviewGET():
	'''
		Find movies by overview (GET request)
		Shows the form for inserting input movie
	'''

	# If the user just wants to acces the site, render the html file
	return render_template('byOverview.html')


@app.route('/byOverview', methods = ['POST'])
def byOverviewPOST():
	'''
		Find movies by overview (POST request)
		Called when the user sents the form
	'''
	
	# If the user has sent the form, process the values
	
	# Obtain movie name, from form
	movie_name = request.form['movie']

	# Obtain results from the model
	try:
		results = prediction.get_recommendations_description(movie_name).to_list()

		# Movie title was found! show the table results!
		return render_template('byOverview.html', 
			recommendedMovies = enumerate(results), 
			numberOfMovies = len(results))

	except: # Movie title was not found
		results = []
		# Do not show the table with the results, but show an error message instead
		return render_template('byOverview.html', 
			recommendedMovies = enumerate(results), 
			numberOfMovies = len(results), 
			MovienotFoundError = True, 
			inputMovie = movie_name)



@app.route('/byGenreAndCast', methods = ['GET'])
def byGenreAndCastGET():
	'''
		Find movies by genre and cast tab (GET request)
	'''

	# If the user just wants to acces the site, render the html file
	return render_template('byGenreAndCast.html')



@app.route('/byGenreAndCast', methods = ['POST'])
def byGenreAndCastPOST():

	'''
		Find movies by genre and cast tab (POST request)
	'''
	
	# If the user has sent the form, process the values
	
	# Obtain movie name, from the form
	movie_name = request.form['movie']

	# Obtain results from the model
	try:
		results = prediction.get_recommendations_plot_cast(movie_name).to_list()

		# Movie title was found! show the table results!
		return render_template('byGenreAndCast.html', 
			recommendedMovies = enumerate(results), 
			numberOfMovies = len(results))

	except: # Movie title was not found
		results = []
		# Do not show the table with the results, but show an error message instead
		return render_template('byGenreAndCast.html', 
			recommendedMovies = enumerate(results), 
			numberOfMovies = len(results), 
			MovienotFoundError = True, 
			inputMovie = movie_name)



@app.route('/login', methods = ['POST', 'GET'])
def login():

	'''
		Login route, shows the form for logging in or showing success / error message if the user logged in 
	'''

	if request.method == 'GET': # Show login site
		return render_template('login.html')
	else: # Has sent login form

		userID = request.form['userID'] # Obtain the ID

		# Validate given input
		if validInputsLoginForm(userID) != 'Valid inputs': # invalid, show error message
			return render_template('login.html', 
				loginError = True, 
				errorMessage = validInputsLoginForm(userID))

		else: # no error, valid user

			userID = int(userID) # cast to integer

		# Check if user exists
		if validateUserID(userID): # user exists
			session['userID'] = userID # create a session with the value of user ID saved to use it
			return render_template('login.html', loginSuccesful = True) # show success message
			
		# User does not exist, show error
		return render_template('login.html', loginError = True, errorMessage = "User does not exists")


@app.route('/logout', methods = ['POST', 'GET'])
def logout():
	'''
		Logout route, allows an user to delete their session an shows the login site
	'''

	session.pop('userID', None)

	return redirect(url_for('login'))


# API requests mode using JSON


@app.route('/recommend_api', methods = ['POST'])
def recommend_api():
	'''
	Allows API requests
	'''

	parameters = request.get_json() # Fetch the received results

	try:
		movieName = parameters['movie'] # Just get the movie title value
	except: # the JSON format is incorrect
		return jsonify({'result' : 'Error!', 'Description': 'The JSON format is incorrect'}) # return JSON response

	try:
		recommendedResults = prediction.get_recommendations_description(movieName).to_list() # get the results from the model
	except: # the movie wasn't found
		return jsonify({'result' : 'Error!', 'Description': 'The movie does not exist'}) # return JSON response

	# Success!
	return jsonify({'result' : 'Success!', 'movie received' : movieName, 'results': recommendedResults}) # return JSON response


if __name__ == '__main__':
	app.run(debug = True)