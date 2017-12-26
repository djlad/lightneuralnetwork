var createNewButtonList = document.getElementsByClassName("component-save-link");
createNewButton = createNewButtonList[0];

var saveButton = document.getElementsByClassName("btn-blue btn-primary wide");
saveButton = saveButton[0];

var saveCurrentAudit = document.getElementsByClassName("btn-blue btn-primary");
saveCurrentAudit = saveCurrentAudit[3];

inputs = document.getElementsByTagName("input");
nameIdBox = inputs[0];
conditionCommentBox = inputs[1];
manufacturerBox = inputs[2];


var selects = document.getElementsByTagName("select");
conditionRatingBox = selects[0];


function insertData(nameId, conditionRating){
    nameIdBox = nameId;
}
