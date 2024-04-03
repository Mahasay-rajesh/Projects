from flask import Flask, render_template, request
import output
import random
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

app = Flask(__name__)   # Flask constructor
  

# Main Page
@app.route('/') 
@app.route('/home')     
def home():

    quotes = ["It is health that is the real wealth, and not pieces of gold and silver",
    "The cheerful mind perseveres, and the strong mind hews its way through a thousand difficulties",
    "I have chosen to be happy because it is good for my health",
    "A sad soul can be just as lethal as a germ",
    "Remain calm, because peace equals power",
    "Healthy citizens are the greatest asset any country can have",
    "Motivation is what gets you started. Habit is what keeps you going",
    "The only bad workout is the one that didn't happen",
    "Challenging yourself every day is one of the most exciting ways to live",
    "When you feel like quitting, think about why you started",
    "The same voice that says 'give up' can also be trained to say 'keep going' "]

    get_quotes = random.sample(quotes, 3)


    return render_template('home.html', val1=get_quotes[0], val2=get_quotes[1], val3=get_quotes[2])    


@app.route('/waterpotability',methods=['GET','POST'])
def waterpotability():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())

        result = output.model8(to_predict_list)
        if result == 1:
            body="portable water"
        elif result == 0:
            body="Non-portable water"
        else:
            body="Invalid Data Entered. Please Enter Numeric Data"

        msg = MIMEMultipart()

        msg['From'] = 'monikaa.adventure@gmail.com'
        msg['To'] = 'monivijayababu@gmail.com'
        msg['Subject'] = 'WATER QUALITY PREDICTION RESULT'

        msg.attach(MIMEText(body))

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login('monikaa.adventure@gmail.com', 'wlcommusiramgkry')
        text = msg.as_string()
        server.sendmail('monikaa.adventure@gmail.com', 'monivijayababu@gmail.com', text)
        server.quit()
            
        return render_template('waterpotability.html',prediction=result)
        
    return render_template('waterpotability.html')

if __name__=='__main__':
   app.run(debug=False)
