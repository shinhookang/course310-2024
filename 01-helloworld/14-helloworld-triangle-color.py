from OpenGL.GL import *
from glfw.GLFW import *

def main():

    # Before you can use most GLFW functions, the library must be initialized.
    # On successful initialization, GLFW_TRUE is returned. If an error occurred, GLFW_FALSE is returned.
    if not glfwInit():
        return 
    
    # Ceate a window and OpenGL context
    window = glfwCreateWindow(640, 480, 'Hello World! a Triangle', None, None)
    if not window:
        glfwTerminate()
        return
    
    # Make the window's context current
    glfwMakeContextCurrent(window)

    # Loop until the user closes the window
    while not glfwWindowShouldClose(window):

        # rendering
        glClear(GL_COLOR_BUFFER_BIT) # clear color buffers to preset values

        # - - - - - - triangle - - - - - 
        glBegin(GL_TRIANGLES)
        glColor3f(1.0, 0.0, 0.0) #R
        glVertex2f(-1/2,-1/2)
        glColor3f(0.0, 1.0, 0.0) #G
        glVertex2f(1/2,-1/2)
        glColor3f(0.0, 0.0, 1.0) #B
        glVertex2f(0,1/2)
        glEnd()
        # - - - - - - triangle - - - - - 

        # Swap front and back buffers
        glfwSwapBuffers(window)

        # Poll for and process events 
        glfwPollEvents()
    
    # When you are done using GLFW, typically just before the application exits, you need to terminate GLFW.
    # This destroys any remaining windows and releases any other resources allocated by GLFW.
    glfwTerminate()

if __name__ == "__main__":
    main()
