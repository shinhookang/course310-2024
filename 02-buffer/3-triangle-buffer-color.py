from OpenGL.GL import *
from glfw.GLFW import *
import numpy as np 

from OpenGL.GL.shaders import compileProgram,compileShader
import ctypes

vertexShaderSrc = '''
#version 330 core
layout (location = 0) in vec3 vertexPosition; 
layout (location = 1) in vec3 vertexColor; 
out vec3 fragmentColor;
void main()
{
    gl_Position = vec4(vertexPosition, 1.0); // w
    fragmentColor = vertexColor;
}
'''

fragmentShaderSrc = '''
#version 330 core
in vec3 fragmentColor;
out vec4 fragmentColorOut;
void main()
{
   fragmentColorOut = vec4(fragmentColor, 1.0); // alpha
}
'''

def CreateShader(vertexShaderSrc, fragmentShaderSrc):

    shader = compileProgram(
        compileShader( vertexShaderSrc, GL_VERTEX_SHADER ),
        compileShader( fragmentShaderSrc, GL_FRAGMENT_SHADER ),
    )

    return shader

def main():

    # Before you can use most GLFW functions, the library must be initialized.
    # On successful initialization, GLFW_TRUE is returned. If an error occurred, GLFW_FALSE is returned.
    if not glfwInit():
        return Exception("Initialization Failed!")
    
    # Ceate a window and OpenGL 3.3 context
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)  
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # only use the modern OpenGL
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) #  request a forward-compatible context for mac OS

    window = glfwCreateWindow(640, 480, 'Hello World! a Triangle', None, None)
    if not window:
        glfwTerminate()
        return Exception("Creating Window Failed!")
    
    # Make the window's context current
    glfwMakeContextCurrent(window)

    # - - - - - - - VBO and VAO - - - - - - - 
    # Triangle vertices and colors
    vertices = [-1/2, -1/2, 0.0,    1.0,  0.0, 0.0, 
                 1/2, -1/2, 0.0,    0.0,  1.0, 0.0,
                 0.0,  1/2, 0.0,    0.0,  0.0, 1.0 ]  
    vertices = np.array(vertices, dtype=np.float32) # 4bytes*18=72bytes

    # Create the name of Vertex Array Object(VAO)
    # Bind a vertex array object to the name of VAO
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao) 

    # Generate the name of Vertex Buffer Object (VBO)
    # Bind a buffer object to the name of VBO
    vbo = glGenBuffers(1) # Error occurs if vbo < 0
    glBindBuffer(GL_ARRAY_BUFFER, vbo) # GL_ARRAY_BUFFER is vertex attribute

    # Copy the vertex data into the target buffer 
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # Specify vertex attribute
    dtype = np.dtype(np.float32).itemsize
    valoc = 0 # the location of the vertex attribute: position
    glVertexAttribPointer(valoc, 3, GL_FLOAT, GL_FALSE, 6*dtype, ctypes.c_void_p(0*dtype)) # vertex data consists of 3 floating points numbers
    glEnableVertexAttribArray(valoc) 
    valoc = 1 # the location of the vertex attribute: color
    glVertexAttribPointer(valoc, 3, GL_FLOAT, GL_FALSE, 6*dtype, ctypes.c_void_p(3*dtype)) # vertex data consists of 3 floating points numbers
    glEnableVertexAttribArray(valoc) 
    
    # - - - - - - - Shader - - - - - - - - - -
    # Note that VAO should be assigned 
    shader = CreateShader(vertexShaderSrc, fragmentShaderSrc)
 
    
    # Loop until the user closes the window
    while not glfwWindowShouldClose(window):

        # rendering
        glClear(GL_COLOR_BUFFER_BIT) # clear color buffers to preset values

        # - - - - - - triangle - - - - - 
        # Draw a triangle with three verticies from array data by starting with the first vertex
        glUseProgram(shader)
        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, 3) 
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
 
