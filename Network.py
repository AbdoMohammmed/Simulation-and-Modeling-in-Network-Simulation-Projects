import simpy
import random
import networkx as nx
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys


# Manual
PACKET_LOSS_PROB = 0.05
CONGESTION_PROB = 0.2

RandomNumber = round(random.uniform(0.0001, 0.003), 5)
packet_size_bits = 1 * 8 * 1024 * 1024 # 1Mb

# Create Topology (red , green)
NETWORK_DELAY = [0.01, 0.03]  # realistic delay
NODE_SPEEDS = {'fast': 0.7, 'slow': 0.3}

class Packet:
    def __init__(self, src, dst, time_sent):
        self.src = src
        self.dst = dst
        self.time_sent = time_sent

def create_graph(n_nodes, n_edges):
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    while len(G.edges) < n_edges:
        u, v = random.sample(range(n_nodes), 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v, cost=random.randint(1, 10) , speed = random.choice([10, 100, 1000]))
    return G

class PrintLogger:
    def __init__(self, text_widget):
        self.text_widget = text_widget
    def write(self, text):
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)
    def flush(self):
        pass

def sender(env, node_id, out_queue, G, counters):
    while True:
        yield env.timeout(random.randint(1, 5))
        dst = random.choice([n for n in G.nodes if n != node_id])
        packet = Packet(src=node_id, dst=dst, time_sent=env.now)
        print(f"[{env.now:.2f}s] Node {node_id} sends packet to Node {dst} with total cost {G[node_id][dst]['cost']} and speed {G[node_id][dst]['speed']:.2f} Gbps")
        out_queue.put(packet)
        counters['sent'] += 1

def router(env, node_id, queue, stats, speed_type, G, packet_count, counters, packet_loss_prob, congestion_prob):
    while True:
        packet = yield queue.get() # Modeling DES
        if random.random() < packet_loss_prob:
            print(f"[{env.now:.2f}s] Packet from {packet.src} to {packet.dst} LOST due to packet loss!")
            counters['lost'] += 1
            continue
        
        try:
            cost = G[node_id][packet.dst].get('weight', 1)
        except KeyError:
            cost = 1  

        delay = random.uniform(*NETWORK_DELAY)
        delay *= cost # Cost >>>>> Delay >>>>>>>>>

        if speed_type == 'slow':
            delay *= 1.5 # Delay >>>>>>>

        if random.random() < congestion_prob:
            congestion_delay = random.uniform(0.03, 0.05)
            print(f"[{env.now:.2f}s] Node {node_id} experiencing CONGESTION! Extra delay of {congestion_delay:.2f}s")
            delay += congestion_delay # Delay >>>>>>>>>>>
            
            if 'congestion_events' not in counters:
                counters['congestion_events'] = []
            counters['congestion_events'].append((env.now, [node_id , packet.dst]))
            
            counters['congestion'] += 1

        yield env.timeout(delay)


        delivery_time = env.now - packet.time_sent # الوقت من اول نود لأخر نود
        print(f"[{env.now:.2f}s] Packet from {packet.src} to {packet.dst} DELIVERED in {delivery_time:.2f}s")
        stats.append(delivery_time) # هنا يسجل الوقت من اول نود لاخر نود
        packet_count[(packet.src, packet.dst)] += 1 # هنا بيعد عدد الباكت
        counters['delivered'] += 1


def run_simulation(n_nodes, n_edges, packet_loss_prob, congestion_prob):
    env = simpy.Environment()
    G = create_graph(n_nodes, n_edges)

    queues = {node: simpy.Store(env) for node in G.nodes}
    stats = []
    packet_count = {(node1, node2): 0 for node1 in G.nodes for node2 in G.nodes}
    node_speeds = {node: 'fast' if random.random() < NODE_SPEEDS['fast'] else 'slow' for node in G.nodes}
    
    counters = {
        'delivered': 0,
        'lost': 0,
        'sent': 0,
        'congestion': 0,
        'congestion_events': []
        
    }

    for node in G.nodes:
        env.process(sender(env, node, queues[node], G, counters))
        env.process(router(env, node, queues[node], stats, node_speeds[node], G, packet_count, counters, packet_loss_prob, congestion_prob))
    
    graph_window = tk.Toplevel()
    graph_window.title("Network Topology")

    fig_topology, ax_topo = plt.subplots(figsize=(6, 5))
    pos = nx.spring_layout(G)
    node_colors = ['green' if node_speeds[node] == 'fast' else 'red' for node in G.nodes]
    
    def draw_topology(highlight_path=None):
        ax_topo.clear()
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=800, font_size=12, ax=ax_topo)
        if highlight_path:
            path_edges = list(zip(highlight_path, highlight_path[1:]))
            nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3, edge_color='purple', ax=ax_topo)
        
        # بعد الحصول على السمات الخاصة بالروابط
        edge_labels = nx.get_edge_attributes(G, 'cost')
        speed_labels = nx.get_edge_attributes(G, 'speed')

        # دمج السرعة والتكلفة في نفس التسميات
        combined_labels = {}
        for edge in edge_labels:
            combined_labels[edge] = f"{speed_labels[edge]} / {edge_labels[edge]}"

        # رسم التسميات المدمجة على الروابط
        nx.draw_networkx_edge_labels(G, pos, edge_labels=combined_labels, ax=ax_topo)
        
        
        ax_topo.set_title("Network Topology")
        canvas_topology.draw()

    canvas_topology = FigureCanvasTkAgg(fig_topology, master=graph_window)
    canvas_topology.draw()
    canvas_topology.get_tk_widget().pack()
    draw_topology()

    def show_best_path_window():
        path_window = tk.Toplevel()
        path_window.title("Best Path Between Two Nodes")
        path_window.geometry("350x350")

        tk.Label(path_window, text="Source Node:", font=("Arial", 14)).pack(pady=10)
        src_entry = tk.Entry(path_window, font=("Arial", 14))  
        src_entry.pack(pady=5)

        tk.Label(path_window, text="Destination Node:", font=("Arial", 14)).pack(pady=10)
        dst_entry = tk.Entry(path_window, font=("Arial", 14))
        dst_entry.pack(pady=5)

        result_label = tk.Label(path_window, text="", font=("Arial", 12), fg="blue")
        result_label.pack(pady=5)
        

        def calculate_best_path():
            try:
           
                src = int(src_entry.get()) # دا اول نود
                dst = int(dst_entry.get()) # دا تاني نود
                
                # التأكد من النود
                if src not in G.nodes or dst not in G.nodes:
                    raise ValueError
                
                # حساب أقصر مسار بين العقد
                path = nx.shortest_path(G, source=src, target=dst, weight='cost') # بيحسب هنا اقصر مسار بال Dijkstra
                total_cost = nx.shortest_path_length(G, source=src, target=dst, weight='cost') # هنا بيجيب الكوست

                
                # إجمالي التأخير بناءً على مسار الرابط
                speeds_along_path = []
                
                for u, v in zip(path[:-1], path[1:]):
                    speeds_along_path.append(G[u][v]['speed']) # مجموع السرعات من النود الاولي للتانية
                
               
                ideal_Max_Link_Speed = min(speeds_along_path) if speeds_along_path else 0 # اقصي حد للسرعة في المسار 
                max_speed_gbps = ideal_Max_Link_Speed - (ideal_Max_Link_Speed * ((PACKET_LOSS_PROB + (RandomNumber * total_cost)))) 
                average_speed = sum(speeds_along_path) / len(speeds_along_path) # متوسط السرعة في المسار
            
            
    
                
                # الوقت اللي بياخده الباكت من اول نود لأخر نود
                total_time_seconds = sum(packet_size_bits / ( (G[u][v]['speed'] - G[u][v]['speed'] * ((PACKET_LOSS_PROB + (RandomNumber * total_cost)))) * 1e9) for u, v in zip(path[:-1], path[1:]))
                total_delay_ms = total_time_seconds * 1000

                result = f"Best Path from Node {src} to Node {dst}:\n{' → '.join(map(str, path))}\nTotal Cost: {total_cost}\n" \
                        f"Ideal Max Link Speed: {ideal_Max_Link_Speed} Gbps\n Actual Max Speed From Source: {max_speed_gbps:.2f} Gbps \n Average Speed: {average_speed} Gbps  \nTotal Delay: {total_delay_ms:.2f} ms"
                
                result_label.config(text=result)
                draw_topology(highlight_path=path)
                
            except Exception as e:
                result_label.config(text="❌ Invalid node IDs or no path exists.")
        
        tk.Button(path_window, text="Calculate", command=calculate_best_path, font=("Arial", 14)).pack(pady=10)

    btn_best_path = tk.Button(graph_window, text="Calculate Best Path", command=show_best_path_window)
    btn_best_path.pack(pady=5)
    
    def show_packet_delivery_stats():
        # حساب عدد الباكتات اللي وصلت واللي وقعت
        delivered = counters['delivered']
        lost = counters['lost']

        # رسم الـ pie chart
        labels = ['Delivered', 'Lost']
        sizes = [delivered, lost]
        colors = ['green', 'red']
        explode = (0.1, 0)  # هنميز الشريحة بتاعة "Delivered" زي ما بيحصل في معظم الرسومات البيانية

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
        ax.axis('equal')  # يضمن إن الـ pie chart يكون دائري مش بيضاوي

        # إنشاء نافذة جديدة لعرض الـ pie chart
        delivery_window = tk.Toplevel()
        delivery_window.title("Packet Delivery Statistics")

        # إضافة Canvas لعرض الـ pie chart في نافذة Tkinter
        canvas = FigureCanvasTkAgg(fig, master=delivery_window)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=20)

        # إضافة نص توضيحي أسفل الـ pie chart
        stats_label = tk.Label(delivery_window, text=f"Delivered: {delivered} packets\nLost: {lost} packets", font=("Arial", 12))
        stats_label.pack(pady=10)
    
    def show_congestion_stats_window():
        congestion_window = tk.Toplevel()
        congestion_window.title("Congestion Delay Statistics")

        congestion_label = tk.Label(congestion_window, text=f"🚧 Congestion Delays Occurred: {counters['congestion']}",
                                    font=("Arial", 12, "bold"), fg="orange")
        congestion_label.pack(pady=10)

        if counters['congestion_events']:
            times = [event[0] for event in counters['congestion_events']]
            links = [f"{pair[0]}→{pair[1]}" for _, pair in counters['congestion_events']]

            fig_congestion, ax_congestion = plt.subplots(figsize=(10, 5))
            ax_congestion.scatter(times, links, color='darkorange', alpha=0.8)
            ax_congestion.set_title("Congestion Events Timeline by Link")
            ax_congestion.set_xlabel("Time (s)")
            ax_congestion.set_ylabel("Link (From → To)")
            ax_congestion.grid(True)

            # تحسين محور X بخطوات زمنية واضحة
            if times:
                min_time = int(min(times))
                max_time = int(max(times))
                interval = 5  # تقدر تغير الرقم حسب كثافة الأحداث
                ax_congestion.set_xticks(range(min_time, max_time + 1, interval))

            fig_congestion.tight_layout()

            canvas_congestion = FigureCanvasTkAgg(fig_congestion, master=congestion_window)
            canvas_congestion.draw()
            canvas_congestion.get_tk_widget().pack(pady=20)
        else:
            no_data_label = tk.Label(congestion_window, text="No congestion events recorded.", font=("Arial", 12))
            no_data_label.pack(pady=20)
            


    btn_congestion_stats = tk.Button(graph_window, text="Show Congestion Stats", command=show_congestion_stats_window)
    btn_congestion_stats.pack(pady=5)
    
    btn_show_packet_stats = tk.Button(graph_window, text="Show Packet Delivery Stats", command=show_packet_delivery_stats)
    btn_show_packet_stats.pack(pady=10)
    


    log_window = tk.Toplevel()
    log_window.title("Simulation Log")
    log_text = scrolledtext.ScrolledText(log_window, width=80, height=20)
    log_text.pack()
    sys.stdout = PrintLogger(log_text)
    env.run(until=50)
    sys.stdout = sys.__stdout__

    graph_stats = tk.Toplevel()
    graph_stats.title("Histograms & Packet Delivery Stats")

    if stats:
        fig_histo, ax = plt.subplots(figsize=(6, 4))  # استخدم محور واحد بدل subplot مزدوج

        # Histogram for delays
        ax.hist(stats, bins=10, color='orange', edgecolor='black')
        avg_delay = sum(stats) / len(stats) if stats else 0
        ax.axvline(avg_delay, color='red', linestyle='dashed', linewidth=2, label=f'Avg = {avg_delay:.3f} s')
        ax.legend()
        ax.set_title("Packet Delivery Times")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Packets")
        ax.grid(True)

        plt.tight_layout()

        canvas_hist = FigureCanvasTkAgg(fig_histo, master=graph_stats)
        canvas_hist.draw()
        canvas_hist.get_tk_widget().pack(pady=10)

        avg_label = tk.Label(
            graph_stats, 
            text=f"📊 Average Delay: {avg_delay:.3f} s", 
            font=("Arial", 12, "bold"), 
            fg="green"
        )
        avg_label.pack(pady=20)

        calc_label = tk.Label(
            graph_stats, 
            text=f"⚙️ Calculated Average Delay: {avg_delay:.3f} s", 
            font=("Arial", 10), 
            fg="black"
        )
        calc_label.pack(pady=10)

    
def main_gui():
    root = tk.Tk()
    root.title("Network Simulation")

    # Adjust window size and position
    window_width = 600
    window_height = 400
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    position_top = int(screen_height / 2 - window_height / 2)
    position_right = int(screen_width / 2 - window_width / 2)
    root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')
    
    tk.Label(root, text="Enter number of nodes:", font=("Arial", 12)).pack(pady=10)
    entry_nodes = tk.Entry(root, font=("Arial", 12))
    entry_nodes.pack()

    tk.Label(root, text="Enter number of edges:", font=("Arial", 12)).pack(pady=10)
    entry_edges = tk.Entry(root, font=("Arial", 12))
    entry_edges.pack()

    tk.Label(root, text="Enter Packet Loss Probability (0-1):", font=("Arial", 12)).pack(pady=10)
    entry_packet_loss = tk.Entry(root, font=("Arial", 12))
    entry_packet_loss.insert(0, str(PACKET_LOSS_PROB))
    entry_packet_loss.pack()

    tk.Label(root, text="Enter Congestion Probability (0-1):", font=("Arial", 12)).pack(pady=10)
    entry_congestion = tk.Entry(root, font=("Arial", 12))
    entry_congestion.insert(0, str(CONGESTION_PROB))
    entry_congestion.pack()

    def start():
        try:
            n = int(entry_nodes.get())
            n_edges = int(entry_edges.get())
            packet_loss = float(entry_packet_loss.get())
            congestion_prob = float(entry_congestion.get())
            if n < 2 or n_edges < 1 or not (0 <= packet_loss <= 1) or not (0 <= congestion_prob <= 1):
                raise ValueError
            global PACKET_LOSS_PROB, CONGESTION_PROB
            PACKET_LOSS_PROB = packet_loss
            CONGESTION_PROB = congestion_prob
            run_simulation(n, n_edges, packet_loss, congestion_prob)
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid values (nodes > 1, edges > 0, speed > 0, probabilities between 0 and 1).")

    tk.Button(root, text="Start Simulation", command=start, font=("Arial", 14)).pack(pady=20)
    root.mainloop()

if __name__ == "__main__":
    main_gui()
