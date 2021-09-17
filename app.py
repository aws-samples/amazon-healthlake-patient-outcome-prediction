#!/usr/bin/env python3

from aws_cdk import core

from patient_outcome_prediction.patient_outcome_prediction_stack import PatientOutcomePredictionStack


app = core.App()
PatientOutcomePredictionStack(app, "patient-outcome-prediction", env={'region': 'us-east-1'})

app.synth()
