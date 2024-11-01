import torch
import os

model = torch.hub.load('ultralytics/yolov5', 'custom',
 
path='C:/Users/Username/PycharmProjects/Yolov5Train/yolov5/runs/train/exp11/weights/best.pt'
)

img = 'C:/Users/Username/Desktop/Validation netlist/TestImage.jpg'

# Get the detection results
results = model(img)

components = []
for *box, conf, cls in results.xyxy[0]: # For each detection
  x1, y1, x2, y2 = box
  component_class = model.names[int(cls)]
  x_center = (x1 + x2) / 2
  y_center = (y1 + y2) / 2
  components.append({'class': component_class, 'x': x_center, 'y': y_center})

counts = {'resistor': 1, 'inductor': 1, 'capacitor': 1, 'dc': 1, 'diode': 1}
counts['ac'] = counts['dc'] + 1

def convert_class_to_netlist(component_class, count):
 if component_class == 'resistor':
   return f'R{count}', '1k'
 elif component_class == 'inductor':
   return f'L{count}', '10m'
 elif component_class == 'capacitor':
   return f'C{count}', '10m'
 elif component_class in ['dc', 'ac']: # Common treatment for DC and AC
   return f'V{count}', '5' if component_class == 'dc' else 'SINE(0 1 1K)'
 elif component_class == 'diode':
   return f'D{count}', 'D'
 return None, None

components = sorted(components, key=lambda c: c['x'])

# Identify the corner components based on sorted positions
left_component = components[0]
right_component = components[-1]
middle_components = sorted(components[1:-1], key=lambda c: c['y'])
top_component = middle_components[0]
bottom_component = middle_components[-1]

# Prepare the netlist
netlist = []
ordered_components = [left_component, top_component, right_component, bottom_component]

current_node = 'N001'
next_node = 'N002'

for comp in ordered_components:
  component_type, default_value = convert_class_to_netlist(comp['class'], counts[comp['class']])
  if comp == left_component:
    node1, node2 = current_node, '0'
    value = input(
      f"{component_type} found between nodes {node1} and {node2}. Please set the value (e.g., 
{default_value}): ")
 elif comp == top_component:
 node1, node2 = next_node, current_node
 value = input(
 f"{component_type} found between nodes {node1} and {node2}. Please set the value (e.g., 
{default_value}): ")
 elif comp == right_component:
   node1, node2 = next_node, 'N003'
   value = input(
     f"{component_type} found between nodes {node1} and {node2}. Please set the value (e.g., 
{default_value}): ")
  elif comp == bottom_component:
   node1, node2 = 'N003', '0'
   value = input(
     f"{component_type} found between nodes {node1} and {node2}. Please set the value (e.g., 
{default_value}): ")
  if comp['class'] == 'ac':
     dc_offset = input(
      f"Alternating Voltage Source {component_type} found between nodes {node1} and 
{node2}. Please set the DC offset (e.g., 0V): ")
   amplitude = input("Please set the Amplitude (e.g., 1V): ")
   frequency = input("Please set the Frequency (e.g., 1kHz): ")
   value = f"SINE({dc_offset} {amplitude} {frequency})"

if comp['class'] in ['ac', 'dc']:
 plus_node = input(
   f"Please set the plus terminal connection of {component_type} (must be N_higher for a 
normal direction of connection): ")
 minus_node = input(
   f"Please set the minus terminal connection of {component_type} (must be N_lower for a 
normal direction of connection): ")
 netlist.append(f"{component_type} {plus_node} {minus_node} {value}")
 
elif comp['class'] == 'diode':
 anode_node = input(
   f"Please set the anode connection of {component_type} (for the normal direction of 
connection must be N_lower): ")
 cathode_node = input(
   f"Please set the cathode connection of {component_type} (for the normal direction of 
connection must be N_higher): ")
 netlist.append(f"{component_type} {anode_node} {cathode_node} {value}")
else:
 netlist.append(f"{component_type} {node1} {node2} {value}")

 counts[comp['class']] += 1

include_op = input("Would you like to include the .op directive in the netlist? (yes/no): ")
if include_op.lower() == 'yes':
 netlist.append(".op")

netlist.append(".model D D")
netlist.append(".lib C:\\Users\\Username\\Documents\\LTspiceXVII\\lib\\cmp\\standard.dio")
netlist.append(".backanno")
netlist.append(".end")

output_file = "C:/Users/Username/Desktop/Validation netlist/Generated.net"
with open(output_file, 'w') as f:
 f.write("* C:\\Users\\Username\\Desktop\\Validation netlist\\Draft1.asc\n")
 for line in netlist:
   f.write(line + '\n')

print(f"Netlist saved to {output_file}"
