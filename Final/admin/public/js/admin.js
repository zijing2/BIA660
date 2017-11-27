(function ($) {

    //prediction
    $("#predict_form").submit(function(){
        return false
    })
    $("#predict_reset").click(function(){
        $("#review").val("")
    });
    $("#predict_submit").click(function(){

        let review = $("#review").val()
        var requestConfig = {
                method: "POST",
                url: "/admin/predict",
                contentType: 'application/json',
                // dataType: 'JSON',
                data: JSON.stringify({
                    "reviews" : [review]
                })
        };
        $.ajax(requestConfig).then(function (resObj) {
                $("#predict_result_content").text(resObj)
                $("#predict_result_content").show()
                $("#predict_result_head").show()
                if(resObj.success==1){
                    debugger
                }else{
                    debugger
                }
        });
    });


})(jQuery);