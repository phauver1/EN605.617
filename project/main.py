# Import the relevant modules
import sys, time, math
import pygame
import cuda_boids
from math import radians, cos
import subprocess

# Define a class for a GUI slider
class Slider:
    # Initialize the slider object
    def __init__(self, rect, min_val, max_val, value, label, to_float=False):
        self.rect = pygame.Rect(rect)
        self.min_val, self.max_val = min_val, max_val
        self.value = value
        self.label = label
        self.dragging = False
        self.to_float = to_float
        self.handle_w = 12

    # Method to set the 'handle' portion of the slider GUI
    def handle_rect(self):
        t = (self.value - self.min_val) / (self.max_val - self.min_val + 1e-8)
        hx = int(self.rect.x + t * (self.rect.w - self.handle_w))
        return pygame.Rect(hx, self.rect.y, self.handle_w, self.rect.h)

    # Draw the full slider GIO
    def draw(self, surf, font):
        # Draw the slider background
        pygame.draw.rect(surf, (60, 60, 80), self.rect, border_radius=6)
        # Compute the amount of fill on the slider based on value
        t = (self.value - self.min_val) / (self.max_val - self.min_val + 1e-8)
        fill_rect = self.rect.copy(); fill_rect.w = int(t * self.rect.w)
        # Draw the filled section of the slider
        pygame.draw.rect(surf, (120, 160, 255), fill_rect, border_radius=6)
        # Draw the handle
        pygame.draw.rect(surf, (220, 220, 250), self.handle_rect(), border_radius=4)
        # Draw the label and the current value
        text = font.render(f"{self.label}: {self.value:.2f}" if self.to_float else f"{self.label}: {self.value}", True, (230,230,240))
        surf.blit(text, (self.rect.x, self.rect.y - 24))

    # Perform actions based on changes to the slider
    def on_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.handle_rect().collidepoint(event.pos) or self.rect.collidepoint(event.pos):
                self.dragging = True; self._update_value_from_mouse(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_value_from_mouse(event.pos)

    # Auxiliary method to pull appropriate value from current mouse position
    def _update_value_from_mouse(self, pos):
        mx = pos[0]
        t = (mx - self.rect.x) / max(1, self.rect.w)
        t = max(0.0, min(1.0, t))
        self.value = self.min_val + t * (self.max_val - self.min_val)
        if not self.to_float: self.value = int(round(self.value))

# Draw the boids as a triangle
def draw_boid(screen, x, y, vx, vy, color=(200,200,255)):
    mag = math.hypot(vx, vy)
    if mag < 1e-5: dirx, diry = 1, 0
    else: dirx, diry = vx/mag, vy/mag
    size = 8; perp_x, perp_y = -diry, dirx
    tip = (int(x + dirx*size), int(y + diry*size))
    left = (int(x - dirx*size*0.5 + perp_x*size*0.5), int(y - diry*size*0.5 + perp_y*size*0.5))
    right = (int(x - dirx*size*0.5 - perp_x*size*0.5), int(y - diry*size*0.5 - perp_y*size*0.5))
    pygame.draw.polygon(screen, color, [tip, left, right])

# Mainloop
def main():
    # Initialize boid parameters
    p = cuda_boids.BoidsParams()
    p.perception_radius = 40.
    p.angle_limit = 120.
    p.max_speed = 80.0
    p.hysteresis = 0.8
    p.align_weight = 1.0
    p.cohese_weight = 1.0
    p.separate_weight = 1.0
    p.boundary_weight = 1.0
    p.world_width = 1000.0
    p.world_height = 700.0

    # Initialize the sim with the default parameters
    sim = cuda_boids.PyBoidsSim(1000, p, 1234)

    # Initialize the GUI window
    pygame.init()
    side_panel_w = 300
    window_w = int(p.world_width) + side_panel_w
    window_h = int(p.world_height)
    screen = pygame.display.set_mode((window_w, window_h))
    pygame.display.set_caption("CUDA Boids")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 18)

    # Define a slider for each of our major settings
    sliders = [
        Slider((int(p.world_width)+20, 40, side_panel_w-40, 16), 100, 2000, int(sim.count()), "Boids"),
        Slider((int(p.world_width)+20, 100, side_panel_w-40, 16), 10, 200, int(p.perception_radius), "Perception radius"),
        Slider((int(p.world_width)+20, 160, side_panel_w-40, 16), 0, 180, int(p.angle_limit), "Angle limit"),
        Slider((int(p.world_width)+20, 220, side_panel_w-40, 16), -10, 10, int(p.align_weight), "Align weight"),
        Slider((int(p.world_width)+20, 280, side_panel_w-40, 16), -10, 10, int(p.cohese_weight), "Cohese weight"),
        Slider((int(p.world_width)+20, 340, side_panel_w-40, 16), -10, 10, int(p.separate_weight), "Separate weight"),
        Slider((int(p.world_width)+20, 400, side_panel_w-40, 16), 10, 250, int(p.max_speed), "Speed"),
        Slider((int(p.world_width)+20, 460, side_panel_w-40, 16), 0., 1., float(p.hysteresis), "Hysteresis", to_float=True),
        Slider((int(p.world_width)+20, 520, side_panel_w-40, 16), -10, 10, int(p.boundary_weight), "Boundary weight"),
    ]

    panel_rect = pygame.Rect(int(p.world_width), 0, side_panel_w, window_h)
    running = True; last_time = time.perf_counter()

    # Set up ffmpeg to save playback video
    ffmpeg = subprocess.Popen([
    'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
    '-s', f'{window_w}x{window_h}', '-r', '60',
    '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', 'boids.mp4'
    ], stdin=subprocess.PIPE)

    while running:
        # Check for events
        for event in pygame.event.get():
            # Exiting the sim
            if event.type == pygame.QUIT:
                running = False
            # Adjusting slider values
            for s in sliders:
                s.on_event(event)

        # Update parameters from sliders
        params = sim.get_params()
        params.perception_radius = float(sliders[1].value)
        params.angle_limit = float(cos(radians(sliders[2].value)))
        params.align_weight = float(sliders[3].value)
        params.cohese_weight = float(sliders[4].value)
        params.separate_weight = float(sliders[5].value)
        params.max_speed = float(sliders[6].value)
        params.hysteresis = float(sliders[7].value)
        params.boundary_weight = float(sliders[8].value)
        sim.set_params(params)

        # Resize boid population if slider changed
        target_n = int(sliders[0].value)
        if target_n != sim.count():
            sim.resize(target_n)

        # Step simulation
        now = time.perf_counter()
        dt = now - last_time
        last_time = now
        sim.step(min(dt, 0.033))

        # Fetch positions and velocities
        pos = sim.positions_host()
        vel = sim.velocities_host()
        N = len(pos) // 2
        xs, ys = pos[:N], pos[N:]
        vxs, vys = vel[:N], vel[N:]

        # Clear screen and draw boids
        screen.fill((12, 14, 22))
        pygame.draw.rect(screen, (20, 22, 32), pygame.Rect(0, 0, int(p.world_width), int(p.world_height)))
        for i in range(N):
            draw_boid(screen, xs[i], ys[i], vxs[i], vys[i])

        # Draw side panel
        pygame.draw.rect(screen, (15, 18, 28), panel_rect)
        title = font.render("Controls", True, (240, 240, 250))
        screen.blit(title, (panel_rect.x + 20, 10))
        for s in sliders:
            s.draw(screen, font)

        # Stats
        fps = clock.get_fps()
        stats = font.render(f"N={sim.count()}  FPS={fps:.1f}", True, (220, 220, 230))
        screen.blit(stats, (panel_rect.x + 20, panel_rect.bottom - 30))

        pygame.display.flip()
        clock.tick(60)

        # Save frame to playback video
        frame_data = pygame.image.tostring(screen, 'RGB')
        ffmpeg.stdin.write(frame_data)

    pygame.quit()

    # Save playback video
    ffmpeg.stdin.close()
    ffmpeg.wait()

if __name__ == "__main__":
    main()
