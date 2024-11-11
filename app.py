# Import necessary modules from Flask for web application functionality
from flask import Flask, render_template, request, url_for, session
# Import numpy for numerical operations
import numpy as np
# Import and set matplotlib backend for plotting
import matplotlib
matplotlib.use("Agg")  # Use "Agg" backend for non-GUI rendering
import matplotlib.pyplot as plt  # Import pyplot for plotting

# Import Linear Regression model from scikit-learn
from sklearn.linear_model import LinearRegression
# Import statistics module from scipy
from scipy import stats

# Initialize Flask app
app = Flask(__name__)
# Set a secret key for session management, replace with your own key for security
app.secret_key = "your_secret_key_here"

# Function to generate data and perform simulations
def generate_data(N, mu, beta0, beta1, sigma2, S):
    # Generate N random values for X
    X = np.random.rand(N)
    # Calculate Y based on linear model with random noise
    Y = beta0 + beta1 * X + mu + np.random.normal(0, np.sqrt(sigma2), N)

    # Fit a linear regression model to data (X, Y)
    model = LinearRegression().fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]  # Extract slope from model
    intercept = model.intercept_  # Extract intercept from model

    # Set up and save scatter plot with regression line
    plot1_path = "static/plot1.png"  # Define path for saving plot
    plt.figure()
    plt.scatter(X, Y, label="Data Points")  # Scatter plot of X and Y
    plt.plot(X, model.predict(X.reshape(-1, 1)), color="red", label="Regression Line")
    plt.xlabel("X")  # Label x-axis
    plt.ylabel("Y")  # Label y-axis
    plt.legend()  # Display legend
    plt.savefig(plot1_path)  # Save plot
    plt.close()  # Close figure to free memory

    # Initialize lists to store slopes and intercepts from simulations
    slopes = []
    intercepts = []
    # Perform S simulations for estimating slopes and intercepts
    for _ in range(S):
        # Generate simulated X values
        X_sim = np.random.rand(N)
        # Generate simulated Y values based on linear model
        Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, np.sqrt(sigma2), N)
        # Fit linear model to simulated data
        sim_model = LinearRegression().fit(X_sim.reshape(-1, 1), Y_sim)
        # Append simulated slope and intercept to lists
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    # Set up and save histograms of simulated slopes and intercepts
    plot2_path = "static/plot2.png"
    plt.figure()
    plt.hist(slopes, bins=20, alpha=0.5, label="Slopes")  # Histogram of slopes
    plt.hist(intercepts, bins=20, alpha=0.5, label="Intercepts")  # Histogram of intercepts
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(plot2_path)
    plt.close()

    # Calculate proportions of slopes and intercepts more extreme than observed values
    slope_more_extreme = np.mean(np.abs(slopes) >= abs(slope))
    intercept_extreme = np.mean(np.abs(intercepts) >= abs(intercept))

    # Return generated data, regression parameters, and plot paths
    return (
        X,
        Y,
        slope,
        intercept,
        plot1_path,
        plot2_path,
        slope_more_extreme,
        intercept_extreme,
        slopes,
        intercepts,
    )

# Route for main page with GET and POST methods
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Retrieve user inputs from the form
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        # Call function to generate data and plots based on user input
        (
            X,
            Y,
            slope,
            intercept,
            plot1,
            plot2,
            slope_extreme,
            intercept_extreme,
            slopes,
            intercepts,
        ) = generate_data(N, mu, beta0, beta1, sigma2, S)

        # Store generated data in session for access in other routes
        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        # Render template with plot paths and calculated values
        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")

# Route for handling "generate" button on webpage
@app.route("/generate", methods=["POST"])
def generate():
    return index()

# Route for performing hypothesis testing
@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    # Retrieve simulation data from session
    N = int(session.get("N", 100))
    S = int(session.get("S", 1000))
    slope = float(session.get("slope", 0))
    intercept = float(session.get("intercept", 0))
    slopes = session.get("slopes", [])
    intercepts = session.get("intercepts", [])
    beta0 = float(session.get("beta0", 0))
    beta1 = float(session.get("beta1", 1))

    # Retrieve test parameters from form
    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    # Select appropriate simulated values based on chosen parameter
    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    # Calculate p-value based on test type (>, <, or !=)
    if test_type == ">":
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "<":
        p_value = np.mean(simulated_stats <= observed_stat)
    else:
        p_value = np.mean(np.abs(simulated_stats - hypothesized_value) >= abs(observed_stat - hypothesized_value))

    # Set a fun message if p-value is very low
    fun_message = "Wow! You've encountered a rare event!" if p_value <= 0.0001 else None

    # Plot histogram of simulated statistics with observed statistic and hypothesized value
    plot3_path = "static/plot3.png"
    plt.figure()
    plt.hist(simulated_stats, bins=20, alpha=0.7, label="Simulated Statistics")
    plt.axvline(observed_stat, color="red", linestyle="--", label="Observed Statistic")
    plt.axvline(hypothesized_value, color="blue", linestyle="--", label="Hypothesized Value")
    plt.xlabel("Statistic Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(plot3_path)
    plt.close()

    # Render template with results and plot path
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        p_value=p_value,
        fun_message=fun_message,
    )

# Route for calculating confidence interval
@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    # Retrieve data for confidence interval calculation
    slopes = np.array(session.get("slopes", []))
    intercepts = np.array(session.get("intercepts", []))
    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level")) / 100

    # Choose correct statistic array based on parameter
    if parameter == "slope":
        stats_array = slopes
        true_value = session.get("beta1", 0)
    else:
        stats_array = intercepts
        true_value = session.get("beta0", 0)
    
    # Verify parameter and confidence level inputs are provided
    parameter = request.form.get("parameter")
    confidence_level_input = request.form.get("confidence_level")

    if parameter is None or confidence_level_input is None:
        return "Error: Parameter or confidence level is not selected.", 400

    # Validate confidence level input as float
    try:
        confidence_level = float(confidence_level_input) / 100
    except ValueError:
        return "Error: Confidence level must be a valid number.", 400

    # Calculate mean and standard error if array is non-empty
    if len(stats_array) > 0:
        mean_estimate = np.mean(stats_array)
        se = np.std(stats_array) / np.sqrt(len(stats_array))
        # Calculate confidence interval using t-distribution
        ci_lower, ci_upper = stats.t.interval(confidence_level, len(stats_array) - 1, loc=mean_estimate, scale=se)
    else:
        mean_estimate, ci_lower, ci_upper = None, None, None

    # Determine if true parameter value lies within confidence interval
    includes_true = (
        "True" if ci_lower is not None and ci_upper is not None and ci_lower <= true_value <= ci_upper else "False"
    )

    # Plot confidence interval with simulated estimates
    plot4_path = "static/plot4.png"
    plt.figure()
    plt.scatter(stats_array, np.zeros_like(stats_array), alpha=0.5, color="gray", label="Simulated Estimates")
    if mean_estimate is not None:
        plt.axvline(mean_estimate, color="blue", linestyle="-", label="Mean Estimate")
    if ci_lower is not None and ci_upper is not None:
        plt.hlines(0, ci_lower, ci_upper, color="green", linestyle="-", label=f"{int(confidence_level * 100)}% Confidence Interval")
    plt.axvline(true_value, color="red", linestyle="--", label="True Value")
    plt.xlabel("Statistic Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig(plot4_path)
    plt.close()

    # Render template with calculated confidence interval details and plot path
    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3="static/plot3.png",
        plot4=plot4_path,
        parameter=parameter,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        confidence_level=int(confidence_level * 100),
    )

# Run app with debug mode enabled for development
if __name__ == "__main__":
    app.run(debug=True)
