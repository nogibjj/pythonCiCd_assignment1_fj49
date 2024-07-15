# Individual Assignment 1: Continuous integration and deployment of python descriptive statistics using pandas [Making use of github actions]

Made by: [Faraz Jawed](https://github.com/farazjawedd)

[![Format](https://github.com/nogibjj/pythonCiCd_assignment1_fj49/actions/workflows/format.yml/badge.svg)](https://github.com/nogibjj/pythonCiCd_assignment1_fj49/actions/workflows/format.yml)
[![Lint](https://github.com/nogibjj/pythonCiCd_assignment1_fj49/actions/workflows/lint.yml/badge.svg)](https://github.com/nogibjj/pythonCiCd_assignment1_fj49/actions/workflows/lint.yml)
[![OnInstall](https://github.com/nogibjj/pythonCiCd_assignment1_fj49/actions/workflows/install.yml/badge.svg)](https://github.com/nogibjj/pythonCiCd_assignment1_fj49/actions/workflows/install.yml)
[![Test](https://github.com/nogibjj/pythonCiCd_assignment1_fj49/actions/workflows/test.yml/badge.svg)](https://github.com/nogibjj/pythonCiCd_assignment1_fj49/actions/workflows/test.yml)

---------
Link to youtube [video](https://youtu.be/ihDVkRadaK0) explaining this
---------
# Summary of the project: 
The workflow involes linting, formatting, testing and downloading all the dependencies required for the project. It's automatically handled by github actions whenever there is a new commit pushed to the repository, to make sure everything works. The badges at the top show the status of the project. 

# Spotify Data Analysis

This Python project analyzes data from the Spotify API, which is stored in a CSV file named `spotify.csv`. It provides insights into song lengths and identifies the top 10 artists with the most chart-topping hits between 2010 and 2022.

## Features

Descriptive statistics on song lengths (in milliseconds) to showcase the variation:

- `Mean = 226033`
- `Median = 221653`
- `Mode = 236133`
- `Std = 42063`

Visualization of the top 10 artists with the most chart-topping hits.

Here is the visualization (also gets saved in the output folder as a png file after deployment is complete):

<img width="1580" alt="Screenshot 2023-09-10 at 7 11 13 PM" src="https://github.com/nogibjj/fj49_week2_ds/assets/101464414/cfc958df-4041-4c8f-be86-ab6885a69074">




## CI/CD Integration

This repository is integrated with a CI/CD template for automatic deployment of Python projects within a virtual environment. 

You can find the template [here] (https://github.com/farazjawedd/python-template-ids706). Feel free to use the template for other projects!

## Development Environment

- The repository includes a `.devcontainer` folder with configurations for the VS Code remote container development environment.
- The `.github/workflows/cicd.yml` file defines the Continuous Integration (CI) workflow using GitHub Actions.

Explore the code and data to gain insights into the world of music with Spotify! 

def categorize_gbp_notional(value):
    if value < 1000000:
        return '< 1M'
    elif value < 2000000:
        return '1M - 2M'
    elif value < 5000000:
        return '2M - 5M'
    elif value < 10000000:
        return '5M - 10M'
    elif value < 20000000:
        return '10M - 20M'
    else:
        return '> 20M'

# Apply the categorization to the carry_trades DataFrame
carry_trades['gbp_notional_range'] = carry_trades['gbpNotional'].apply(categorize_gbp_notional)
