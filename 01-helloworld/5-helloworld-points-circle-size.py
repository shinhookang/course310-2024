from OpenGL.GL import *
from glfw.GLFW import *

import numpy as np

def main():

    # Before you can use most GLFW functions, the library must be initialized.
    # On successful initialization, GLFW_TRUE is returned. If an error occurred, GLFW_FALSE is returned.
    if not glfwInit():
        return 
    
    # Ceate a window and OpenGL context
    window = glfwCreateWindow(640, 480, 'Hello World! Points of increasing size on a circle', None, None)
    if not window:
        glfwTerminate()
        return
    
    # Make the window's context current
    glfwMakeContextCurrent(window)

    # Loop until the user closes the window
    while not glfwWindowShouldClose(window):

        # rendering
        glClear(GL_COLOR_BUFFER_BIT) # clear color buffers to preset values

        # - - - - - - points - - - - - 
        n = 12
        h = 2*np.pi/n
        r = 0.5
        for i in range(n):
            psize = 30-2*i
            glPointSize(psize)

            theta = h*i
            xi = r*np.cos(theta)
            yi = r*np.sin(theta)
            glBegin(GL_POINTS)
            glVertex2f(xi,yi)
            glEnd()
        # - - - - - - points - - - - - 

        # Swap front and back buffers
        glfwSwapBuffers(window)

        # Poll for and process events 
        glfwPollEvents()
    
    # When you are done using GLFW, typically just before the application exits, you need to terminate GLFW.
    # This destroys any remaining windows and releases any other resources allocated by GLFW.
    glfwTerminate()

if __name__ == "__main__":
    main()
