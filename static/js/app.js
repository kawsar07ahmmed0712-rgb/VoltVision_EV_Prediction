const form =
    document.getElementById(
        "predictionForm"
    );


const submitButton =
    document.getElementById(
        "submitButton"
    );


const demoButton =
    document.getElementById(
        "demoButton"
    );


const formError =
    document.getElementById(
        "formError"
    );


const concernInput =
    document.getElementById(
        "Environmental_Concern_Level"
    );


const concernValue =
    document.getElementById(
        "concernValue"
    );


const emptyState =
    document.getElementById(
        "emptyState"
    );


const resultState =
    document.getElementById(
        "resultState"
    );


const resultBadge =
    document.getElementById(
        "resultBadge"
    );


const scoreRing =
    document.getElementById(
        "scoreRing"
    );


const scoreValue =
    document.getElementById(
        "scoreValue"
    );


const tierLabel =
    document.getElementById(
        "tierLabel"
    );


const resultTitle =
    document.getElementById(
        "resultTitle"
    );


const resultText =
    document.getElementById(
        "resultText"
    );


const modelElements = {

    lightgbm: {
        bar:
            document.getElementById(
                "lgbBar"
            ),

        value:
            document.getElementById(
                "lgbValue"
            ),
    },


    xgboost: {
        bar:
            document.getElementById(
                "xgbBar"
            ),

        value:
            document.getElementById(
                "xgbValue"
            ),
    },


    catboost: {
        bar:
            document.getElementById(
                "catBar"
            ),

        value:
            document.getElementById(
                "catValue"
            ),
    },
};


const numericFields = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
];


/* ============================================================
   Concern Slider
   ============================================================ */

concernInput.addEventListener(

    "input",

    () => {

        concernValue.textContent =
            concernInput.value;
    }
);


/* ============================================================
   Utility: set select only when option exists
   ============================================================ */

function setSelectValue(
    id,
    preferredValues
) {

    const element =
        document.getElementById(
            id
        );


    if (!element) {

        return;
    }


    const availableValues =
        Array.from(
            element.options
        ).map(
            option =>
                option.value
        );


    for (
        const value
        of preferredValues
    ) {

        if (
            availableValues.includes(
                value
            )
        ) {

            element.value =
                value;

            return;
        }
    }


    if (
        element.options.length
        >
        0
    ) {

        element.selectedIndex =
            0;
    }
}


/* ============================================================
   Clamp Demo Numeric Values to Input Range
   ============================================================ */

function setNumericValue(
    id,
    desiredValue
) {

    const input =
        document.getElementById(
            id
        );


    if (!input) {

        return;
    }


    let value =
        Number(
            desiredValue
        );


    const minimum =
        Number(
            input.min
        );


    const maximum =
        Number(
            input.max
        );


    if (
        input.min
        !==
        ""
        &&
        Number.isFinite(
            minimum
        )
    ) {

        value =
            Math.max(
                minimum,
                value
            );
    }


    if (
        input.max
        !==
        ""
        &&
        Number.isFinite(
            maximum
        )
    ) {

        value =
            Math.min(
                maximum,
                value
            );
    }


    input.value =
        value;
}


/* ============================================================
   Demo Profile
   ============================================================ */

demoButton.addEventListener(

    "click",

    () => {


        hideError();


        setNumericValue(
            "Age",
            44
        );


        setNumericValue(
            "Annual_Income_USD",
            105000
        );


        setNumericValue(
            "Daily_Commute_km",
            24.5
        );


        setNumericValue(
            "Number_of_Cars_Owned",
            2
        );


        setNumericValue(
            "Charging_Stations_Near_Home",
            6
        );


        setNumericValue(
            "Charging_Stations_Near_Work",
            9
        );


        setNumericValue(
            "Environmental_Concern_Level",
            5
        );


        concernValue.textContent =
            concernInput.value;


        setSelectValue(
            "Gender",
            [
                "Female",
                "Male",
            ]
        );


        setSelectValue(
            "City_Type",
            [
                "Suburban",
                "Urban",
                "Rural",
            ]
        );


        setSelectValue(
            "Current_Car_Type",
            [
                "SUV",
                "Sedan",
                "Hatchback",
                "Truck",
            ]
        );


        setSelectValue(
            "Home_Charging_Possible",
            [
                "Yes",
                "No",
            ]
        );


        setSelectValue(
            "Range_Anxiety_Level",
            [
                "Low",
                "Medium",
                "High",
            ]
        );


        const yesSubsidy =
            form.querySelector(
                'input[name="Subsidy_Available"][value="Yes"]'
            );


        if (yesSubsidy) {

            yesSubsidy.checked =
                true;
        }

    }
);


/* ============================================================
   Form Submission
   ============================================================ */

form.addEventListener(

    "submit",

    async event => {


        event.preventDefault();


        hideError();


        if (
            !form.checkValidity()
        ) {

            form.reportValidity();

            return;
        }


        const payload =
            collectPayload();


        setLoading(
            true
        );


        try {


            const response =
                await fetch(

                    "/predict",

                    {
                        method:
                            "POST",

                        headers: {
                            "Content-Type":
                                "application/json",

                            "Accept":
                                "application/json",
                        },

                        body:
                            JSON.stringify(
                                payload
                            ),
                    }
                );


            let data;


            try {

                data =
                    await response.json();

            }

            catch {

                throw new Error(
                    "The server returned an invalid response."
                );
            }


            if (
                !response.ok
                ||
                !data.ok
            ) {

                throw new Error(
                    data.error
                    ||
                    "Unable to generate a prediction."
                );
            }


            renderResult(
                data
            );


        }

        catch (
            error
        ) {


            showError(
                error.message
                ||
                "Something went wrong. Please try again."
            );


        }

        finally {


            setLoading(
                false
            );

        }

    }
);


/* ============================================================
   Collect Payload
   ============================================================ */

function collectPayload() {


    const data =
        Object.fromEntries(

            new FormData(
                form
            ).entries()
        );


    numericFields.forEach(

        field => {

            data[field] =
                Number(
                    data[field]
                );

        }
    );


    return data;
}


/* ============================================================
   Loading
   ============================================================ */

function setLoading(
    state
) {


    submitButton.disabled =
        state;


    submitButton.classList.toggle(
        "loading",
        state
    );
}


/* ============================================================
   Errors
   ============================================================ */

function showError(
    message
) {


    formError.textContent =
        message;


    formError.classList.add(
        "visible"
    );


    formError.scrollIntoView(
        {
            behavior:
                "smooth",

            block:
                "nearest",
        }
    );
}


function hideError() {


    formError.textContent =
        "";


    formError.classList.remove(
        "visible"
    );
}


/* ============================================================
   Result Rendering
   ============================================================ */

function renderResult(
    data
) {


    emptyState.classList.add(
        "hidden"
    );


    resultState.classList.remove(
        "hidden"
    );


    resultBadge.classList.remove(
        "idle"
    );


    resultBadge.classList.add(
        "active"
    );


    resultBadge.textContent =
        `${data.tier} propensity`;


    tierLabel.textContent =
        `${data.tier} propensity`;


    resultTitle.textContent =
        getResultTitle(
            data.tier
        );


    resultText.textContent =
        getResultDescription(
            data.tier
        );


    animateScore(
        Number(
            data.percentage
        )
    );


    renderModelProbability(
        "lightgbm",
        data.models.lightgbm
    );


    renderModelProbability(
        "xgboost",
        data.models.xgboost
    );


    renderModelProbability(
        "catboost",
        data.models.catboost
    );


    if (
        window.innerWidth
        <
        1150
    ) {


        document.querySelector(
            ".result-card"
        ).scrollIntoView(
            {
                behavior:
                    "smooth",

                block:
                    "start",
            }
        );

    }

}


/* ============================================================
   Individual Model Bar
   ============================================================ */

function renderModelProbability(
    modelName,
    probability
) {


    const element =
        modelElements[
            modelName
        ];


    if (!element) {

        return;
    }


    const percentage =

        Math.max(
            0,

            Math.min(
                100,

                Number(
                    probability
                )
                *
                100
            )
        );


    element.value.textContent =
        `${percentage.toFixed(1)}%`;


    element.bar.style.width =
        "0%";


    requestAnimationFrame(
        () => {

            requestAnimationFrame(
                () => {

                    element.bar.style.width =
                        `${percentage}%`;

                }
            );

        }
    );

}


/* ============================================================
   Main Score Animation
   ============================================================ */

function animateScore(
    target
) {


    const finalValue =

        Math.max(
            0,

            Math.min(
                100,
                Number(
                    target
                )
            )
        );


    const duration =
        900;


    const started =
        performance.now();


    function update(
        now
    ) {


        const progress =

            Math.min(

                (
                    now
                    -
                    started
                )

                /

                duration,

                1
            );


        const eased =

            1

            -

            Math.pow(
                1
                -
                progress,

                3
            );


        const current =
            finalValue
            *
            eased;


        scoreValue.textContent =
            `${current.toFixed(1)}%`;


        scoreRing.style.setProperty(

            "--score-angle",

            `${current * 3.6}deg`
        );


        if (
            progress
            <
            1
        ) {

            requestAnimationFrame(
                update
            );
        }

    }


    requestAnimationFrame(
        update
    );
}


/* ============================================================
   Dynamic Result Copy
   ============================================================ */

function getResultTitle(
    tier
) {


    const titles = {

        Low:
            "Low EV purchase signal",

        Moderate:
            "Moderate EV purchase signal",

        Elevated:
            "Elevated EV purchase signal",

        High:
            "Strong EV purchase signal",

    };


    return (
        titles[tier]
        ||
        "EV purchase propensity"
    );
}


function getResultDescription(
    tier
) {


    const descriptions = {

        Low:

            "The profile currently contains relatively weak signals associated with EV purchase intent.",


        Moderate:

            "The customer shows a mixed combination of signals, resulting in moderate purchase propensity.",


        Elevated:

            "Multiple customer and infrastructure signals align with stronger EV purchase intent.",


        High:

            "The ensemble detects a strong combination of economic, behavioral and infrastructure signals associated with EV purchase intent.",

    };


    return (
        descriptions[tier]
        ||
        "The score combines predictions from LightGBM, XGBoost and CatBoost."
    );
}