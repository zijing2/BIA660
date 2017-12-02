(function ($) {

    //prediction
    $("#predict_form").submit(function(){
        return false
    })
    $("#predict_reset").click(function(){
        $("#review").val("")
    });

    $("#predict_submit").click(function(){
        $("#predict_submit").attr("class", "templatemo-blue-button btn disabled")
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
                $("#predict_submit").attr("class", "templatemo-blue-button")
                var res_json = JSON.parse(resObj)
                $("#predict_result_content").text(resObj)
                // $("#predict_result_content").show()
                // $("#predict_result_head").show()
                $("#predict_div").html(res_json.html)
                if(resObj.success==1){
                    //debugger
                }else{
                    //debugger
                }
        });
    });


})(jQuery);