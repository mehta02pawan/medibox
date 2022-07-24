from datetime import datetime, timedelta
from cal_setup import get_calendar_service

def main(drug_dose, prescription_date,drg):
   # creates one hour event tomorrow 10 AM IST
   service = get_calendar_service()

   today = datetime(prescription_date.year, prescription_date.month, prescription_date.day, 10)
   start = today.isoformat()
   end = (today + timedelta(days=int(drug_dose[0][1]))).isoformat()

   body = {
           "summary": 'take your ',
           "description": 'trial',
           "start": {"dateTime": start, "timeZone": 'Asia/Kolkata'},
           "end": {"dateTime": end, "timeZone": 'Asia/Kolkata'},
           'recurrence': ['RRULE:FREQ=DAILY;COUNT='+str(drug_dose[0][0])],
       }
   service.events().insert(calendarId='primary', body=body).execute()

def loop(drug_dose, prescription_date,drg,desc):
   # creates one hour event tomorrow 10 AM IST
   service = get_calendar_service()
   for i in range(len(drug_dose)):
       today = datetime(prescription_date.year, prescription_date.month, prescription_date.day, 10)
       start = today.isoformat()
       end = (today + timedelta(days=int(drug_dose[i][1]))).isoformat()

       body = {
               "summary": 'take your '+str(drg[i]),
               "description": 'Time to take your medication',
               "start": {"dateTime": start, "timeZone": 'GMT-4:00'},
               "end": {"dateTime": end, "timeZone": 'GMT-4:00'},
               'recurrence': ['RRULE:FREQ=DAILY;COUNT='+str(drug_dose[i][0])],
           }
       service.events().insert(calendarId='primary', body=body).execute()


   # print("created event")
   # print("id: ", event_result['id'])
   # print("summary: ", event_result['summary'])
   # print("starts at: ", event_result['start']['dateTime'])
   # print("ends at: ", event_result['end']['dateTime'])

# if __name__ == '__main__':
#    main(qty,duration, prescription_date)
