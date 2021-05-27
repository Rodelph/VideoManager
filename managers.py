import cv2
import numpy as np
import time

"""
We create a class called CaptureManager and WindowManager as high-level interfaces to I/O streams. This applicaiton code may use CaptureManager to read
new frames and optionally, to dispatch each frame to one or more outputs, including a still image file, and a window (wia a WindowManager class). 
A WindowManager class lets our application code handle a window and events in an object-oriented style.

A CaptureManager object is initialized with a VideoCapture object and has enterFrame and exitFrame methods that should typically be called on every iteration 
of an application's main loop. Between a call to enterFrame and exitFrame, the application may (any number of times) set a channel property and get a frame
property is an image corresponding to the current channel's state when enterFrame was called. 

A CaptureManager also has the writeImage, startWritingVideo, and stopWritingVideo methods that may be called at any time. Actual file writing is postponed 
until next frame may be shown in a window, depending on whether the application code provides a WindowManager class either as an argument to the constructor 
of CaptureManager or by setting the previewWindowManager property. 

If the application code manipulates frame, the manipulations are reflected in recorded files and in the window. A CaptureManager class has a constructor argument 
and property called shouldMirrorPreview, which should be True if we want frame to be mirrored (horizontally flipped) in the window but not in recorded files. Typically, 
when facing a camera.

Recall that a VideoWriter object needs a frame rate, but OpenCV does not provide any reliable way to get an accurate frame rate for a camera. The CaptureManager class 
works around this limitation by using a frame counter and Python's standard time.time function to estimate the frame rate if necessary. This approach is not foolproof. 
Depending on frame rate fluctuations and the system-dependent implementation of time.time, the accuracy of the estimate might still be poor in some cases. However, 
if we deploy to unknown hardware, it is better than just assuming that the user's camera has a particular frame rate.
"""


class CaptureManager(object):

    def __init__(self, capture, previewWindowManager=None, shouldMirrorPreview=False, shouldConvertBitDepth10To8=True):

        self.previewWindowManager = previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview
        self.shouldConvertBitDepth10To8 = \
            shouldConvertBitDepth10To8
        self._capture = capture  # Non public variable
        self._channel = 0  # Non public variable
        self._enteredFrame = False  # Non public variable
        self._frame = None  # Non public variable
        self._imageFilename = None  # Non public variable
        self._videoFilename = None  # Non public variable
        self._videoEncoding = None  # Non public variable
        self._videoWriter = None  # Non public variable
        self._startTime = None  # Non public variable
        self._framesElapsed = 0  # Non public variable
        self._fpsEstimate = None  # Non public variable

    # Adding getters and setters

    # We use non public variables to relate to the state of the current frame and any file-writing ioperation. The code only needs to configure few things, which are
    # implemented as constructor arguments and settable public properties:
    #    - The camera channel
    #    - The window manager
    #    - The option to mirror the camera preview

    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self, value):
        if self._channel != value:
            self._channel = value
            self._frame = None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            _, self._frame = self._capture.retrieve(
                self._frame, self.channel)
            # The second if statement will help us manipulate and display frames form some channels, notably cv2.CAP_OPENNI_IR_IMAGE.
            if self.shouldConvertBitDepth10To8 and \
                    self._frame is not None and \
                    self._frame.dtype == np.uint16:
                self._frame = (self._frame >> 2).astype(np.uint8)
        return self._frame

    @property
    def isWritingImage(self):
        return self._imageFilename is not None

    @property
    def isWritingVideo(self):
        return self._videoFilename is not None

    def enterFrame(self):
        """Capture the next frame, if any."""

        # But first, check that any previous frame was exited.
        assert not self._enteredFrame, \
            'previous enterFrame() had no matching exitFrame()'

        if self._capture is not None:
            self._enteredFrame = self._capture.grab()

        # The implementation fo exitFrame takes the image from the current channel, estimates a frame rate, shows the image via the window manager

    # (if any), and fulfills any pending request to write the image to files.

    def exitFrame(self):
        """Draw to the window. Write to files. Release the frame."""

        # Check whether any grabbed frame is retrievable.
        # The getter may retrieve and cache the frame.
        if self.frame is None:
            self._enteredFrame = False
            return

        # Update the FPS estimate and related variables.
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() - self._startTime
            self._fpsEstimate = self._framesElapsed / timeElapsed
        self._framesElapsed += 1

        # Draw to the window, if any.
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = np.fliplr(self._frame)
                self.previewWindowManager.show(mirroredFrame)
            else:
                self.previewWindowManager.show(self._frame)

        # Write to the image file, if any.
        if self.isWritingImage:
            cv2.imwrite(self._imageFilename, self._frame)
            self._imageFilename = None

        # Write to the video file, if any.
        self._writeVideoFrame()

        # Release the frame.
        self._frame = None
        self._enteredFrame = False

    # The following methods 'writeImage', 'startWritingImage', and 'stopWritingImage' simply update the parameters for file-writing operations,
    # whereas the actual writing operations are postponed to the next call of exitFrame.

    def writeImage(self, filename):
        """Write the next exited frame to an image file."""
        self._imageFilename = filename

    def startWritingVideo(
            self, filename, encoding=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')):
        """Start writing exited frames to a video file."""
        self._videoFilename = filename
        self._videoEncoding = encoding

    def stopWritingVideo(self):
        """Stop writing exited frames to a video file."""
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

        if self._videoWriter is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0.0:
                # The capture's FPS is unknown so use an estimate.
                if self._framesElapsed < 20:
                    # Wait until more frames elapse so that the
                    # estimate is more stable.
                    return
                else:
                    fps = self._fpsEstimate
            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._videoWriter = cv2.VideoWriter(self._videoFilename, self._videoEncoding, fps, size)

        self._videoWriter.write(self._frame)

    # The following method creates or appends to a video file. However, in situations where the frame rate is unkown, we skip some frames at the start of the capture session
    # so that we have time to build up the estimate of the frame rate.

    def _writeVideoFrame(self):

        if not self.isWritingVideo:
            return

        if self._videoWriter is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps <= 0.0:
                # The capture's FPS is unknown so use an estimate.
                if self._framesElapsed < 20:
                    # Wait until more frames elapse so that the
                    # estimate is more stable.
                    return
                else:
                    fps = self._fpsEstimate
            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self._videoWriter = cv2.VideoWriter(
                self._videoFilename, self._videoEncoding, fps, size)

        self._videoWriter.write(self._frame)


# For the sake of object orientation and adaptability, we abstract this functionality into a WindowManager class with the createWindow,
# destroyWindow, show and processEvents methods. As a property, WindowManager has a function object called keypressCallback, which (if it is not 'None')
# is called from processEvents in response to any keypress. The keypressCallback object is a function that takes a single argument, specifically an ASCII keycode.

class WindowManager(object):
    def __init__(self, windowName, keypressCallback=None):
        self.keypressCallback = keypressCallback
        self._windowName = windowName
        self._isWindowCreated = False

    @property
    def isWindowCreated(self):
        return self._isWindowCreated

    def createWindow(self):
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):
        cv2.imshow(self._windowName, frame)

    def destroyWindow(self):
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvents(self):
        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode != -1:
            self.keypressCallback(keycode)