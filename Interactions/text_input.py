# Copyright (c) 2022 Grumpy Cat Software S.L.
#
# This Source Code is licensed under the MIT 2.0 license.
# the terms can be found in LICENSE.md at the root of
# this project, or at http://mozilla.org/MPL/2.0/.

import shapelets as sh
sh.login(user_name='admin',password='admin')

from shapelets.apps import DataApp, TextInput

def change_text(text: str) -> str:
    return text

def change_text1(text: str) -> TextInput:
    return TextInput(value=text)

# Create a dataApp
app = DataApp(name="text_input")

# Create a text input
text = app.text_input(title="Greeting", value="Hey Joe!")

# Bind to avoid change in text
text.bind(change_text, "hey! don't change me", triggers=[text])

# Place text input into the Dataapp
app.place(text)

# Create a text_input
text1 = app.text_input()

# Bind
text1.bind(change_text1, "Setting an alternative text",triggers=[text1])

# Place text with title into the Dataapp
app.place(text1)


# Create a text input with default value
text2 = app.text_input(title="This is a text with default text", 
    placeholder="Ponga un texto aquí")
# Place text input with default text into the Dataapp
app.place(text2)

# Create text input
textStyle = {"color": 'blue', "width": 150, "fontSize": 28,
             "fontFamily": 'Papyrus', "fontStyle": 'italic',  "fontWeight": 'bold'}
text3 = app.text_input(title="This text input is an entry parameter",
                       value="Text to show", 
                       placeholder="Entry parameter text", 
                       multiline=True, text_style=textStyle, markdown=True)
app.place(text3)

app.register()
