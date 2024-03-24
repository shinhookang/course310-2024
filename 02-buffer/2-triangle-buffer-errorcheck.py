from OpenGL.GL import *
from glfw.GLFW import *
import numpy as np 

vertexShaderSrc = '''
#version 330 core
layout (location = 0) in vec3 vertexPosition; 
out vec3 fragmentColor;
void main()
{
    gl_Position = vec4(vertexPosition, 1.0); // w
    fragmentColor = vec3(1.0, 0.0, 0.0);
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

    # vertex shader 
    vertexShader = glCreateShader(GL_VERTEX_SHADER) # creates an empty shader object and returns a non-zero value
    glShaderSource(vertexShader, vertexShaderSrc)   # replaces the source code in a shader object
    glCompileShader(vertexShader)                   # compile the shader object
    
    success = glGetShaderiv(vertexShader, GL_COMPILE_STATUS)
    if (not success):
        Exception("Compilation of the vertex shader failed!")
        
    # fragment shader
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER) # creates an empty shader object and returns a non-zero value
    glShaderSource(fragmentShader, fragmentShaderSrc) # replaces the source code in a shader object
    glCompileShader(fragmentShader)                   # compile the shader object

    success = glGetShaderiv(fragmentShader, GL_COMPILE_STATUS)
    if (not success):
        Exception("Compilation of the fragment shader failed!")
        
    # link shaders
    shader = glCreateProgram()               # creates an empty program object and returns a non-zero value
    glAttachShader(shader, vertexShader)     
    glAttachShader(shader, fragmentShader)
    glLinkProgram(shader)                    # links a program object

    success = glGetProgramiv(shader, GL_LINK_STATUS)
    if (not success):
        Exception("Program linking failed!")
        
    glDeleteShader(vertexShader)
    glDeleteShader(fragmentShader)

    return shader
    

def main():

    # Before you can use most GLFW functions, the library must be initialized.
    # On successful initialization, GLFW_TRUE is returned. If an error occurred, GLFW_FALSE is returned.
    if not glfwInit():
        return Exception("Initialization Failed!")
    
    # Ceate a window and OpenGL context
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # only use the modern OpenGL
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) #  request a forward-compatible context for mac OS

    window = glfwCreateWindow(640, 480, 'Hello World! a Triangle', None, None)
    if not window:
        glfwTerminate()
        return Exception("Creating Window Failed!")
    
    # Make the window's context current
    glfwMakeContextCurrent(window)

    # - - - - - - - Shader - - - - - - - - - -
    shader = CreateShader(vertexShaderSrc, fragmentShaderSrc)
 
    
    # - - - - - - - VBO and VAO - - - - - - - 
    # Triangle vertices 
    vertices = [-1/2, -1/2, 0.0, # v0
                 1/2, -1/2, 0.0, # v1
                 0.0,  1/2, 0.0] # v2
    vertices = np.array(vertices, dtype=np.float32) # 4bytes*9=36bytes
     
    
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
    valoc = 0 # the location of the vertex attribute
    glVertexAttribPointer(valoc, 3, GL_FLOAT, GL_FALSE, 3*np.dtype(np.float32).itemsize, None) # vertex data consists of 3 floating points numbers
    glEnableVertexAttribArray(valoc) 
    
    

    # Loop until the user closes the window
    while not glfwWindowShouldClose(window):

        # clear color buffers to preset values
        glClear(GL_COLOR_BUFFER_BIT) 

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
    
    # delete buffer
    glDeleteBuffers(1,vbo)
    glDeleteVertexArrays(1,vao)
    
    # When you are done using GLFW, typically just before the application exits, you need to terminate GLFW.
    # This destroys any remaining windows and releases any other resources allocated by GLFW.
    glfwTerminate()

if __name__ == "__main__":
    main()
