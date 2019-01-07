import sys
sys.path.append('/home/pi/lcd')
import lcd
lcd.GPIO.setwarnings(False) 
lcd.lcd_init()
lcd.lcd_byte(lcd.LCD_LINE_1, lcd.LCD_CMD)
lcd.lcd_string("Raspberry Pi", 2)
lcd.lcd_byte(lcd.LCD_LINE_2, lcd.LCD_CMD)
lcd.lcd_string("Model B+", 2)
lcd.GPIO.cleanup()
