#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import sklearn
import boto3
import joblib


def handler(event, context):

    model = joblib.load('./lr_credit_risk_model.pki')
    credit_amount = int(event.get('credit_amount'))
    age = int(event.get('age'))
    duration = int(event.get('duration'))

    value_to_predict = [[credit_amount, #Credit amount
                        age, #Age in years,
                        duration, #Duration in month,
                        1, #Status of existing checking account_0 <= <200 SGD,
                        0, #Status of existing checking account_<0 SGD,
                        0, #Status of existing checking account_>= 200 SGD,
                        0, #Status of existing checking account_no checking account,
                        0, #Credit history_all credits at this bank paid back duly,
                        1, #Credit history_critical account,
                        0, #Credit history_delay in paying off,
                        0, #Credit history_existing credits paid back duly till now,
                        0, #Credit history_no credits taken,
                        1, #Purpose_business,
                        0, #Purpose_car (new),
                        0, #Purpose_car (used),
                        0, #Purpose_domestic appliances,
                        1, #Purpose_education,
                        0, #Purpose_furniture/equipment,
                        0, #Purpose_others,
                        0, #Purpose_radio/television,
                        0, #Purpose_repairs,
                        0, #Purpose_retraining,
                        0, #Savings account/bonds_100 <= <500 SGD,
                        0, #Savings account/bonds_500 <= < 1000 SGD,
                        0, #Savings account/bonds_<100 SGD,
                        0, #Savings account/bonds_>= 1000 SGD,
                        1, #Savings account/bonds_no savings account,
                        0, #Present employment since_1<= < 4 years,
                        0, #Present employment since_4<= <7 years,
                        0, #Present employment since_<1 years,
                        0, #Present employment since_>=7 years,
                        1, #Present employment since_unemployed,
                        0, #Personal status and sex_female:divorced/separated/married,
                        0, #Personal status and sex_male:divorced/separated,
                        1, #Personal status and sex_male:married/widowed,
                        0, #Personal status and sex_male:single,
                        0, #Property_car or other,
                        0, #Property_real estate,
                        1, #Property_savings agreement/life insurance,
                        0, #Property_unknown / no property,
                        1, #Other installment plans_bank,
                        0, #Other installment plans_none,
                        0, #Other installment plans_store,
                        0, #Housing_for free,
                        0, #Housing_own,
                        1, #Housing_rent,
                        1, #foreign worker_no,
                        0 #foreign worker_yes
                        ]]
    predicted_risk_values = model.predict(value_to_predict)  

    print(predicted_risk_values[0])
    #print(sklearn.__version__)
    # TODO implement
    risk={1:"Good Risk", 0:"Bad Risk"}
    print("This customer is a {} customer".format(risk.get(predicted_risk_values[0])))
    
    return {
        'statusCode': 200,
        'body': json.dumps(
              {  'Result': risk.get(predicted_risk_values[0]) })
            }