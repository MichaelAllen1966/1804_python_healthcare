import simpy
import random
import pandas as pd
import matplotlib.pyplot as plt

class g():
    # Global variables
    bed_count=0
    inter_arrival_time=1
    los=10
    sim_duration=500
    audit_time=[]
    audit_beds=[]
    audit_interval=1

def new_admission(env,interarrival_time,los):
    i=0
    while True:
        i+=1
        p_los=random.expovariate(1/los)
        p=patient(env,i,p_los)
        env.process(p)
        next_p=random.expovariate(1/interarrival_time)
        # print('Next patient in %f3.2' %next_p)
        yield env.timeout(next_p)

def patient(env,i,p_los):
    g.bed_count+=1
    # print('Patient %d arriving %7.2f, bed count %d' %(i,env.now,g.bed_count))
    yield env.timeout(p_los)
    g.bed_count-=1
    # print('Patient %d leaving %7.2f, bed count %d' %(i,env.now,g.bed_count))

def audit_beds(env,delay):
    yield env.timeout(delay)
    while True:
        g.audit_time.append(env.now)
        g.audit_beds.append(g.bed_count)
        yield env.timeout(g.audit_interval)

def build_audit_report():
    audit_report=pd.DataFrame()
    audit_report['Time']=g.audit_time
    audit_report['Occupied_beds']=g.audit_beds
    audit_report['Median_beds']=audit_report['Occupied_beds'].quantile(0.5)
    audit_report['Beds_5_percent']=audit_report['Occupied_beds'].quantile(0.05)
    audit_report['Beds_95_percent']=audit_report['Occupied_beds'].quantile(0.95)
    return audit_report

def chart():
    plt.plot(g.audit_dataframe['Time'],g.audit_dataframe['Occupied_beds'],color='k',marker='o',linestyle='solid',markevery=1,label='Occupied beds')
    plt.plot(g.audit_dataframe['Time'],g.audit_dataframe['Beds_5_percent'],color='0.5',linestyle='dashdot',markevery=1,label='5th percentile')
    plt.plot(g.audit_dataframe['Time'],g.audit_dataframe['Median_beds'],color='0.5',linestyle='dashed',label='Median')
    plt.plot(g.audit_dataframe['Time'],g.audit_dataframe['Beds_95_percent'],color='0.5',linestyle='dashdot',label='95th percentile')
    plt.xlabel('Day')
    plt.ylabel('Occupied beds') 
    plt.title('Occupied beds (individual days with 5th, 50th and 95th percentiles)')
    #plt.legend()
    plt.show()


def main():
    # Initialise environment
    env=simpy.Environment()
    # Initialise processes (admissions & bed audit)
    env.process(new_admission(env,g.inter_arrival_time,g.los))
    env.process(audit_beds(env,delay=20))
    # Start simulation run
    env.run(until=g.sim_duration)
    # Build audit table
    g.audit_dataframe=build_audit_report()
    chart()


if __name__=='__main__':
    main()
