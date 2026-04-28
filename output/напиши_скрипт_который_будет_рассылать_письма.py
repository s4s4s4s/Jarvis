import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def load_subscriber_list(file_path):
    subscribers = []
    with open(file_path, 'r') as file:
        for line in file:
            subscribers.append(line.strip())
    return subscribers

def send_email(subject, body, to_email):
    from_email = "your-email@example.com"
    from_password = "yourpassword"

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.example.com', 587)
    server.starttls()
    server.login(from_email, from_password)
    text = msg.as_string()
    server.sendmail(from_email, to_email, text)
    server.quit()

# Example usage:
subscribers = load_subscriber_list('subscribers.txt')
for subscriber in subscribers:
    send_email("Hello", "This is the body of the email.", subscriber)
