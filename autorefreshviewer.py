#
# autorefreshviewer.py
#
# python script to display a .png file given on the command line in a
# gtk window, and monitor the file for changes, automatically reloading
# and refreshing the view when the file changes.
#
# note that to avoid the file being overwritten while we are in the
# middle of reading from it, we first rename the file, then wait for the
# original file to reappear (meaning that the next frame is being written)
# when it reappears, we load the renamed file (since we know it is no
# longer being written to) and then rename the new file (the one that just
# appeared and is now being written to) for it to finish being written.
# this clearly won't work if whatever you're using to create the file
# opens and closes it multiple times when writing or for some other reason
# expects that the filename not change until it is done writing.
#
# hopefully that wasn't too confusing or anything.

# usage: python3 pngview.py somefile.png

import sys
import time
import os
from PIL import Image
import gi
gi.require_version('Gtk', '3.0')
gi.require_version('GLib', '2.0')
from gi.repository import Gtk, GdkPixbuf, GLib
import pathlib
import shutil
import threading

class ImageViewer(Gtk.Window):
		def __init__(self, filename, destfilename, delayticks=3):
				Gtk.Window.__init__(self)
				self.connect("destroy", Gtk.main_quit)
				self.set_default_size(400, 400)
				self.set_title(filename)
				self.image = Gtk.Image()
				self.add(self.image)
				self.show_all()
				filename = pathlib.Path(filename)
				destfilename = pathlib.Path(destfilename)
				#self.load(destfilename)
				# run monitor in a thread
				self.thread = threading.Thread(target=self.monitor, args=(filename, destfilename))
				self.thread.start()
				#self.monitor(filename, destfilename)

		def load(self, fromfile):
				try:
						# non thread-safe way to load image:
						#self.pixbuf = GdkPixbuf.Pixbuf.new_from_file(str(fromfile))
						#self.image.set_from_pixbuf(self.pixbuf)

						# thread-safe way to load image and update gui in main thread:
						print("loading image from %s" % fromfile)
						pixbuf = GdkPixbuf.Pixbuf.new_from_file(str(fromfile))
						print("loaded image from %s" % fromfile)
						def update_image(pixbuf):
								print("updating image in main thread")
								self.image.set_from_pixbuf(pixbuf)
								print("updated image in main thread")
						print("scheduling image update in main thread")
						GLib.idle_add(update_image, pixbuf)
						print("scheduled image update in main thread")
				except Exception as e:
						print("error loading file: %s" % e)
						return False
				return True

		def monitor(self, filename, destfilename, delayticks=3):
				# self.last_mtime = os.path.getmtime(filename)
				tick = 0
				lastticks = 0
				notloaded = True
				loaderror = False
				while True:
						threading.Event().wait(1)

						tick = tick + 1
						pbar = '...   ...'[tick - (tick//6)*6:][:3]
						print(f"Checking for image{pbar}", end="\r")

						if (filename.exists() or tick - lastticks > delayticks) and destfilename.exists():
							if self.load(destfilename):
								loaderror = False
								notloaded = False
								print("loaded image from %s, removing" % destfilename)
								try:
										if filename.exists():
											os.remove(destfilename)
											loaderror = False
								except Exception as e:
										print("error removing file: %s" % e)
										loaderror = True
							else:
									print("error loading image from %s" % destfilename)
									loaderror = True

						if (notloaded or loaderror or not destfilename.exists()) and filename.exists():
							print("new image detected, moving to %s" % destfilename)
							try:
									os.rename(str(filename), str(destfilename))
									lastticks = tick
									notloaded = True
							except Exception as e:
									print("error moving file: %s" % e)
									return False
						# mtime = os.path.getmtime(filename)
						#if mtime != self.last_mtime:
						#   self.load(destfilename)
						#    self.last_mtime = mtime

import click, pathlib
@click.command()
@click.argument("filename", type=str)
@click.option("-t", "--tempfile", type=str, default=None)
def main(filename, tempfile):
		import pathlib
		filepath = pathlib.Path(filename)
		if tempfile != None:
			destfilename = tempfile
		else:
			destfilename = filepath.parent / f".{filepath.stem}.temp{filepath.suffix}"
		viewer = ImageViewer(str(filepath), str(destfilename))
		Gtk.main()

if __name__ == "__main__":
	main()
