import json


def writeJson(field,val):
    with open('serverupdates.json', 'r') as f:
        current_data = json.load(f)
            
    current_data[field] = val
            
            # FIXED INDENTATION BELOW
    with open("serverupdates.json", 'w') as f:
        json.dump(current_data, f, indent=4) # Now properly indented
                
def readJson(field):
    with open('serverupdates.json', 'r') as f:
        current_data = json.load(f)
    return current_data[field]
            
def triggerDoor():
    if readJson("DoorState") == "open":
        writeJson("command", "ACTIVATE")
    time.sleep(5)
    writeJson("command", "NONE")