# Deploying a Dash Application on Render

This repository will guide you through the process of deploying a Python Dash application on Render using a Github remote repository.

## Prerequisites
1. A Render account. If you do not have one, you can create it by signing up at [https://render.com/](https://render.com/).
2. A Github account and a repository containing the Python Dash application project.

## Step 1: Preparing the Github repository for deployment
### Setting up `app.py`
1. Locate the section of code where the `app` variable is defined. It should be something like the following:

```
app = dash.Dash(__name__)
```

2. Add the following line of code after the `app` variable definition:

```
server = app.server
```

### Creating/updating `requirements.txt`
1. Create a new file called `requirements.txt` in the root directory of the repository. This file should contain all the dependency libraries required for your app to run. For example, in our application:

```
dash==2.9.3
numpy
pandas
scikit_learn
plotly
```

**Note:** Versions of libraries should be omitted (except the version of Dash ) in `requirements.txt` when deploying on Render as the latest version of the platform library will be automatically installed. This ensures that the application is always running on the latest stable version of the library and reduces the chances of compatibility issues arising from different library versions between the local environment and the deployment environment.

2. Add `Gunicorn` library to `requirements.txt`. 

```
gunicorn
```

**Note:** Gunicorn is a Python HTTP server widely used to deploy Python applications including Flask and Django. It provides a scalable, reliable, and fast way to serve web applications and can handle multiple requests simultaneously, which helps improve the performance and stability of web applications.

## Step 2: Setting up a new web service on Render
1. Go to [Render Dashboard](https://dashboard.render.com/) and create a new web service.
2. Provide the public GitHub repository URL for your Dash application.
3. Once you have selected the repository, Render will detect the application type and will prompt you to configure the service. For our application, you need to specify a unique name for the web service and update the start command in the  "Start Command" field. The "Start Command" should be updated as follows:

```
$ gunicorn app:server
```

4. Click the "Create Web Service" button to create the service.

## Step 3: Deploying the service
* After you configure the service, click the "New Deploy" button to deploy the service.
* Render will automatically build and deploy your application. This may take several minutes to complete.
* Once the deployment is complete, you can click on the service URL to view your application.

Our [dashboard application](https://wine-quality-dashboard-app.onrender.com) is now successfully deployed on Render. It is now live and can be accessed from anywhere with an internet connection.








