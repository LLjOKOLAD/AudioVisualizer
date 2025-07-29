import numpy as np
import pygame
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, messagebox
import queue
import threading
import json
import os

try:
    import time
except ImportError as e:
    print(f"Ошибка импорта time: {e}")
    raise


class AudioVisualizer:
    def __init__(self):
        # Параметры по умолчанию
        self.CHUNK = 2048  # Размер буфера
        self.RATE = 48000  # Частота дискретизации
        self.BARS = 50  # Количество баров
        self.WIDTH, self.HEIGHT = 800, 600  # Размер окна
        self.SCALE = 2  # Чувствительность
        self.AUTO_SCALE = False  # Автоматическая подстройка
        self.MIN_BAR_HEIGHT = 10  # Минимальная высота бара
        self.GAIN = 2.0  # Усиление входного сигнала
        self.FFT_GAIN = 10.0  # Усиление FFT амплитуд
        self.MAX_BAR_HEIGHT = 0.8  # Максимальная высота бара
        self.DECAY_FACTOR = 0.5  # Коэффициент угасания баров
        self.BORDER_RADIUS = 0  # Скругление углов баров
        self.USE_CAPS = False  # Использовать крышки
        self.CAP_DECAY_FACTOR = 0.9  # Коэффициент угасания крышек
        self.DEBUG_OUTPUT = True  # Отладочный вывод

        # Путь к конфигурации
        self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

        # Цвета для баров (RGB)
        self.COLORS = {
            "Циан": (0, 255, 255),
            "Красный": (255, 0, 0),
            "Зелёный": (0, 255, 0),
            "Синий": (0, 0, 255),
            "Жёлтый": (255, 255, 0),
            "Радужные": None
        }

        # Глобальные переменные
        self.current_bars = self.BARS
        self.current_scale = self.SCALE
        self.current_color = self.COLORS["Циан"]
        self.current_device = None
        self.borderless = False
        self.last_scale = self.SCALE
        self.decay_factor = self.DECAY_FACTOR
        self.border_radius = self.BORDER_RADIUS
        self.use_caps = self.USE_CAPS
        self.cap_decay_factor = self.CAP_DECAY_FACTOR
        self.debug_output = self.DEBUG_OUTPUT
        self.last_magnitudes = None
        self.last_cap_heights = None
        self.screen_width, self.screen_height = None, None
        self.screen = None
        self.console = None
        self.running = True  # Флаг для управления потоком

        # Инициализация Pygame
        try:
            pygame.init()
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.RESIZABLE)
            pygame.display.set_caption("Аудиовизуализатор (Микрофон)")
            self.screen_width, self.screen_height = pygame.display.Info().current_w, pygame.display.Info().current_h
            if self.debug_output:
                print(f"Разрешение экрана: {self.screen_width}x{self.screen_height}")
        except Exception as e:
            print(f"Ошибка инициализации Pygame: {e}")
            raise

    def hsv_to_rgb(self, h, s=1, v=1):
        import colorsys
        v = min(v, 1.0)  # Ограничиваем яркость для HSV
        return tuple(int(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))

    def analyze_audio(self, data):
        try:
            raw_amplitude = np.max(np.abs(data))
            if self.debug_output:
                print(f"Сырая амплитуда: {raw_amplitude:.2f}")

            fft_data = np.fft.fft(data * self.GAIN)
            fft_magnitude = np.abs(fft_data)[:self.CHUNK // 4] * self.FFT_GAIN
            fft_magnitude = np.log1p(fft_magnitude)  # Мягкое логарифмическое масштабирование
            bar_width = len(fft_magnitude) // self.current_bars
            bar_magnitudes = []
            for i in range(0, len(fft_magnitude), bar_width):
                if len(bar_magnitudes) < self.current_bars:
                    bar_magnitudes.append(np.mean(fft_magnitude[i:i + bar_width]))
            while len(bar_magnitudes) < self.current_bars:
                bar_magnitudes.append(0)

            # Подавление первых 3 баров
            for i in range(min(3, len(bar_magnitudes))):
                bar_magnitudes[i] *= 0.5

            # Угасание амплитуд баров
            if self.last_magnitudes is None or len(self.last_magnitudes) != self.current_bars:
                self.last_magnitudes = [0] * self.current_bars
            decayed_magnitudes = []
            for i, magnitude in enumerate(bar_magnitudes):
                if i < len(self.last_magnitudes) and magnitude < self.last_magnitudes[i]:
                    magnitude = max(magnitude, self.last_magnitudes[i] * (1 - self.decay_factor))
                decayed_magnitudes.append(magnitude)
            self.last_magnitudes = decayed_magnitudes

            # Угасание крышек
            if self.use_caps:
                if self.last_cap_heights is None or len(self.last_cap_heights) != self.current_bars:
                    self.last_cap_heights = [0] * self.current_bars
                cap_heights = []
                for i, magnitude in enumerate(decayed_magnitudes):
                    if magnitude > self.last_magnitudes[i] * (1 - self.decay_factor):  # Если бар растёт
                        cap_height = 10  # Фиксированная высота крышки
                    else:
                        cap_height = max(0, self.last_cap_heights[i] * (1 - self.cap_decay_factor))  # Угасание
                    cap_heights.append(cap_height)
                self.last_cap_heights = cap_heights
            else:
                self.last_cap_heights = [0] * self.current_bars

            if self.debug_output:
                print(f"Амплитуды первых 5 баров (с угасанием): {[f'{x:.2f}' for x in decayed_magnitudes[:5]]}")
                if self.use_caps:
                    print(f"Высота первых 5 крышек: {[f'{x:.2f}' for x in self.last_cap_heights[:5]]}")

            if self.AUTO_SCALE:
                max_amplitude = max(decayed_magnitudes, default=1)
                if max_amplitude > 0:
                    target_scale = max(0.1, min(5, max_amplitude / (self.screen.get_height() / 5)))
                    self.current_scale = 0.5 * self.last_scale + 0.5 * target_scale
                    self.last_scale = self.current_scale
                if self.debug_output:
                    print(f"Макс. FFT амплитуда: {max_amplitude:.2f}, Чувствительность: {self.current_scale:.2f}")

            return decayed_magnitudes
        except Exception as e:
            print(f"Ошибка в analyze_audio: {e}")
            self.console.insert(tk.END, f"Ошибка в analyze_audio: {e}\n") if self.console else None
            return [0] * self.current_bars

    def visualize(self, bars):
        try:
            self.screen.fill((0, 0, 0))
            bar_width = self.screen.get_width() // self.current_bars
            max_height = self.screen.get_height() * self.MAX_BAR_HEIGHT
            for i, magnitude in enumerate(bars):
                height = int(magnitude * 50 / self.current_scale) + self.MIN_BAR_HEIGHT
                height = min(height, int(max_height))
                if self.current_color is None:
                    hue = i / self.current_bars
                    bar_color = self.hsv_to_rgb(hue)
                    cap_color = self.hsv_to_rgb(hue, s=1, v=1.2)  # Более яркий для крышек
                else:
                    bar_color = self.current_color
                    cap_color = (255, 255, 255)  # Белый для крышек при однотонных барах
                # Отрисовка бара
                pygame.draw.rect(self.screen, bar_color,
                                 (i * bar_width, self.screen.get_height() - height, bar_width - 2, height),
                                 border_radius=self.border_radius)
                # Отрисовка крышки с зазором
                if self.use_caps and i < len(self.last_cap_heights):
                    cap_height = int(self.last_cap_heights[i])
                    cap_height = min(cap_height, int(max_height * 0.1))  # Ограничение высоты крышки
                    if cap_height > 0:
                        pygame.draw.rect(self.screen, cap_color, (
                        i * bar_width, self.screen.get_height() - height - cap_height - 5, bar_width - 2, cap_height),
                                         border_radius=self.border_radius)
                        if self.debug_output:
                            print(f"Крышка {i}: высота={cap_height}, цвет={cap_color}")
            pygame.display.flip()
        except Exception as e:
            print(f"Ошибка в visualize: {e}")
            self.console.insert(tk.END, f"Ошибка в visualize: {e}\n") if self.console else None

    def microphone_source(self, data_queue, device_index):
        try:
            device_name = sd.query_devices(device_index)['name']
            print(f"Запуск визуализации с устройства {device_index}: {device_name}")
            self.console.insert(tk.END, f"Запуск визуализации с устройства {device_name}\n") if self.console else None
            with sd.InputStream(device=device_index, samplerate=self.RATE, channels=1, blocksize=self.CHUNK,
                                callback=lambda indata, frames, time, status: data_queue.put(
                                        indata[:, 0]) if not status else print(status), latency='low'):
                while self.running:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                            break
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_F11:
                                self.borderless = not self.borderless
                                try:
                                    if self.borderless:
                                        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height),
                                                                              pygame.NOFRAME)
                                        pygame.display.set_caption("Аудиовизуализатор (Безрамочный)")
                                        print(
                                            f"Переключение в безрамочный режим: {self.screen_width}x{self.screen_height}")
                                        self.console.insert(tk.END,
                                                            f"Switched to borderless mode: {self.screen_width}x{self.screen_height}\n")
                                    else:
                                        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT),
                                                                              pygame.RESIZABLE)
                                        pygame.display.set_caption("Аудиовизуализатор (Микрофон)")
                                        print(f"Переключение в оконный режим: {self.WIDTH}x{self.HEIGHT}")
                                        self.console.insert(tk.END,
                                                            f"Switched to windowed mode: {self.WIDTH}x{self.HEIGHT}\n")
                                except Exception as e:
                                    print(f"Ошибка переключения режима: {e}")
                                    self.console.insert(tk.END, f"Ошибка переключения режима: {e}\n")
                            elif event.key == pygame.K_ESCAPE and self.borderless:
                                self.borderless = False
                                try:
                                    self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.RESIZABLE)
                                    pygame.display.set_caption("Аудиовизуализатор (Микрофон)")
                                    print(f"Выход из безрамочного режима: {self.WIDTH}x{self.HEIGHT}")
                                    self.console.insert(tk.END,
                                                        f"Switched to windowed mode: {self.WIDTH}x{self.HEIGHT}\n")
                                except Exception as e:
                                    print(f"Ошибка выхода из безрамочного режима: {e}")
                                    self.console.insert(tk.END, f"Ошибка выхода из безрамочного режима: {e}\n")
                        elif event.type == pygame.VIDEORESIZE and not self.borderless:
                            try:
                                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                                print(f"Новое разрешение окна: {event.w}x{event.h}")
                                self.console.insert(tk.END, f"Window resized to: {event.w}x{event.h}\n")
                            except Exception as e:
                                print(f"Ошибка изменения размера окна: {e}")
                                self.console.insert(tk.END, f"Ошибка изменения размера окна: {e}\n")

                    try:
                        if not data_queue.empty():
                            data = data_queue.get_nowait()
                            bars = self.analyze_audio(data)
                            self.visualize(bars)
                        time.sleep(0.02)
                    except queue.Empty:
                        pass
                    except Exception as e:
                        print(f"Ошибка обработки данных: {e}")
                        self.console.insert(tk.END, f"Ошибка обработки данных: {e}\n")
                        break
        except Exception as e:
            print(f"Ошибка при захвате звука с микрофона: {e}")
            self.console.insert(tk.END, f"Ошибка микрофона: {e}\n")
        finally:
            print("Завершение визуализации")
            self.console.insert(tk.END, "Визуализация завершена\n") if self.console else None

    def save_config(self):
        config = {
            'device': self.current_device,
            'bars': self.current_bars,
            'scale': self.current_scale,
            'auto_scale': self.AUTO_SCALE,
            'color': [k for k, v in self.COLORS.items() if
                      v == self.current_color or (k == "Радужные" and self.current_color is None)][0],
            'decay_factor': self.decay_factor,
            'border_radius': self.border_radius,
            'use_caps': self.use_caps,
            'cap_decay_factor': self.cap_decay_factor,
            'debug_output': self.debug_output
        }
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            print(f"Настройки сохранены в {self.config_file}: {config}")
            if self.console:
                self.console.insert(tk.END, f"Configuration saved to {self.config_file}\n")
        except Exception as e:
            print(f"Ошибка сохранения конфигурации в {self.config_file}: {e}")
            if self.console:
                self.console.insert(tk.END, f"Failed to save config: {str(e)}\n")

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self.current_device = config.get('device', None)
                    self.current_bars = config.get('bars', self.BARS)
                    self.current_scale = config.get('scale', self.SCALE)
                    self.AUTO_SCALE = config.get('auto_scale', self.AUTO_SCALE)
                    color_name = config.get('color', 'Циан')
                    self.current_color = self.COLORS[color_name] if color_name != "Радужные" else None
                    self.decay_factor = config.get('decay_factor', self.DECAY_FACTOR)
                    self.border_radius = config.get('border_radius', self.BORDER_RADIUS)
                    self.use_caps = config.get('use_caps', self.USE_CAPS)
                    self.cap_decay_factor = config.get('cap_decay_factor', self.CAP_DECAY_FACTOR)
                    self.debug_output = config.get('debug_output', self.DEBUG_OUTPUT)
                    print(f"Настройки загружены из {self.config_file}: {config}")
                    if self.console:
                        self.console.insert(tk.END, f"Configuration loaded from {self.config_file}\n")
            except Exception as e:
                print(f"Ошибка загрузки конфигурации из {self.config_file}: {e}")
                if self.console:
                    self.console.insert(tk.END, f"Failed to load config: {str(e)}\n")
        else:
            print(f"Файл {self.config_file} не найден, используются настройки по умолчанию")
            if self.console:
                self.console.insert(tk.END, f"Configuration file {self.config_file} not found, using defaults\n")

    def create_gui(self):
        try:
            root = tk.Tk()
            root.title("Настройки визуализатора")
            root.geometry("300x700")

            # Консоль для вывода сообщений
            self.console = tk.Text(root, height=5, width=30)
            self.console.pack(pady=5)

            # Загрузка настроек
            self.load_config()

            tk.Label(root, text="Выберите микрофон:").pack(pady=5)
            try:
                devices = sd.query_devices()
                input_devices = [(i, dev['name']) for i, dev in enumerate(devices) if dev['max_input_channels'] > 0]
                if self.debug_output:
                    print("Доступные устройства ввода:", input_devices)
            except Exception as e:
                self.console.insert(tk.END, f"Не удалось получить список устройств: {e}\n")
                root.destroy()
                return

            if not input_devices:
                self.console.insert(tk.END, "Микрофоны не найдены. Проверьте настройки звука.\n")
                root.destroy()
                return

            device_names = [name for _, name in input_devices]
            self.device_var = tk.StringVar(value=device_names[0] if device_names else "")
            if self.current_device is not None:
                try:
                    self.device_var.set(next(name for i, name in input_devices if i == self.current_device))
                except StopIteration:
                    pass
            device_menu = ttk.Combobox(root, textvariable=self.device_var, values=device_names, state="readonly")
            device_menu.pack(pady=5)

            tk.Label(root, text="Количество баров (10-100):").pack(pady=5)
            self.bars_entry = tk.Entry(root)
            self.bars_entry.insert(0, str(self.current_bars))
            self.bars_entry.pack(pady=5)

            tk.Label(root, text="Чувствительность (0.1-5):").pack(pady=5)
            self.scale_entry = tk.Entry(root)
            self.scale_entry.insert(0, str(self.current_scale))
            self.scale_entry.pack(pady=5)

            self.auto_scale_var = tk.BooleanVar(value=self.AUTO_SCALE)
            tk.Checkbutton(root, text="Автоматическая чувствительность", variable=self.auto_scale_var).pack(pady=5)

            tk.Label(root, text="Цвет баров:").pack(pady=5)
            self.color_var = tk.StringVar(value="Циан")
            if self.current_color is None:
                self.color_var.set("Радужные")
            else:
                for k, v in self.COLORS.items():
                    if v == self.current_color:
                        self.color_var.set(k)
                        break
            color_menu = ttk.Combobox(root, textvariable=self.color_var, values=list(self.COLORS.keys()),
                                      state="readonly")
            color_menu.pack(pady=5)

            tk.Label(root, text="Плавность угасания баров (0.0-0.9):").pack(pady=5)
            self.decay_scale = tk.Scale(root, from_=0.0, to=0.9, resolution=0.1, orient=tk.HORIZONTAL,
                                        command=lambda val: setattr(self, 'decay_factor', float(val)))
            self.decay_scale.set(self.decay_factor)
            self.decay_scale.pack(pady=5)

            tk.Label(root, text="Скругление баров (0-20):").pack(pady=5)
            self.radius_scale = tk.Scale(root, from_=0, to=20, resolution=1, orient=tk.HORIZONTAL,
                                         command=lambda val: setattr(self, 'border_radius', int(val)))
            self.radius_scale.set(self.border_radius)
            self.radius_scale.pack(pady=5)

            tk.Label(root, text="Использовать крышки баров:").pack(pady=5)
            self.cap_var = tk.BooleanVar(value=self.use_caps)
            tk.Checkbutton(root, text="Включить крышки", variable=self.cap_var).pack(pady=5)

            tk.Label(root, text="Плавность угасания крышек (0.0-0.9):").pack(pady=5)
            self.cap_decay_scale = tk.Scale(root, from_=0.0, to=0.9, resolution=0.1, orient=tk.HORIZONTAL,
                                            command=lambda val: setattr(self, 'cap_decay_factor', float(val)))
            self.cap_decay_scale.set(self.cap_decay_factor)
            self.cap_decay_scale.pack(pady=5)

            tk.Label(root, text="Отладочный вывод:").pack(pady=5)
            self.debug_var = tk.BooleanVar(value=self.debug_output)
            tk.Checkbutton(root, text="Включить отладочный вывод", variable=self.debug_var,
                           command=lambda: setattr(self, 'debug_output', self.debug_var.get())).pack(pady=5)

            def start_visualizer():
                try:
                    self.current_bars = int(self.bars_entry.get())
                    if not 10 <= self.current_bars <= 100:
                        raise ValueError("Количество баров должно быть от 10 до 100")
                    if not self.auto_scale_var.get():
                        self.current_scale = float(self.scale_entry.get())
                        if not 0.1 <= self.current_scale <= 5:
                            raise ValueError("Чувствительность должна быть от 0.1 до 5")
                    self.AUTO_SCALE = self.auto_scale_var.get()
                    self.use_caps = self.cap_var.get()
                    self.debug_output = self.debug_var.get()
                    color_name = self.color_var.get()
                    self.current_color = self.COLORS[color_name] if color_name != "Радужные" else None
                    device_name = self.device_var.get()
                    self.current_device = next(i for i, name in input_devices if name == device_name)
                    self.save_config()
                    data_queue = queue.Queue()
                    threading.Thread(target=self.microphone_source, args=(data_queue, self.current_device),
                                     daemon=True).start()
                except ValueError as e:
                    self.console.insert(tk.END, f"Ошибка: {str(e)}\n")
                except Exception as e:
                    self.console.insert(tk.END, f"Ошибка при выборе устройства: {str(e)}\n")

            tk.Button(root, text="Запустить визуализатор", command=start_visualizer).pack(pady=10)

            def toggle_borderless():
                self.borderless = not self.borderless
                try:
                    if self.borderless:
                        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.NOFRAME)
                        pygame.display.set_caption("Аудиовизуализатор (Безрамочный)")
                        print(f"Переключение в безрамочный режим: {self.screen_width}x{self.screen_height}")
                        self.console.insert(tk.END,
                                            f"Switched to borderless mode: {self.screen_width}x{self.screen_height}\n")
                    else:
                        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.RESIZABLE)
                        pygame.display.set_caption("Аудиовизуализатор (Микрофон)")
                        print(f"Переключение в оконный режим: {self.WIDTH}x{self.HEIGHT}")
                        self.console.insert(tk.END, f"Switched to windowed mode: {self.WIDTH}x{self.HEIGHT}\n")
                except Exception as e:
                    print(f"Ошибка переключения режима: {e}")
                    self.console.insert(tk.END, f"Ошибка переключения режима: {e}\n")

            tk.Button(root, text="Безрамочный режим (F11, Esc)", command=toggle_borderless).pack(pady=10)

            root.mainloop()
        except Exception as e:
            print(f"Ошибка в create_gui: {e}")
            self.console.insert(tk.END, f"Не удалось создать GUI: {e}\n") if self.console else None


def main():
    try:
        app = AudioVisualizer()
        app.create_gui()
    except Exception as e:
        print(f"Ошибка в main: {e}")
        messagebox.showerror("Ошибка", f"Критическая ошибка: {e}")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()