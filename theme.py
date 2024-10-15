JS_LIGHT_THEME = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontWeight = 'bold';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';

    var text = 'E-commerce FAQ Chatbot 🤖';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 1s';
                letter.innerText = text[i];

                container.appendChild(letter);

                setTimeout(function() {
                    letter.style.opacity = '1';
                }, 50);
            }, i * 125);
        })(i);
    }

    var gradioContainer = document.querySelector('.gradio-container');
    gradioContainer.insertBefore(container, gradioContainer.firstChild);

    return 'Animation created';
}
"""

CSS = """
.btn {
    background-color: #64748B;
    color: #FFFFFF;
}

# #stop_btn {
#     background-color: #ff7373;
#     color: #000000;
# }

#language .selected {
    background-color: #223d5e;
    color: #FFFFFF;
}

#submit_btn {
    background-color: #223d5e;
    color: #FFFFFF;
}

#upload_btn_img {
    background-color: #223d5e;
    color: #FFFFFF;

}#upload_btn_txt {
    background-color: #223d5e;
    color: #FFFFFF;

}#upload_btn_pkl {
    background-color: #223d5e;
    color: #FFFFFF;
}

"""
