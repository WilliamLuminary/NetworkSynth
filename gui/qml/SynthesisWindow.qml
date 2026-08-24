import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts

ApplicationWindow {
    id: root
    visible: true
    width: 720
    height: 780
    title: "NetworkSynth — Synthesis"

    // `controller` is exposed from Python as a context property.

    header: ToolBar {
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 12
            anchors.rightMargin: 12
            Label {
                text: "Synthetic network generation"
                font.pixelSize: 16
                font.bold: true
                Layout.fillWidth: true
            }
            Label {
                text: "Mode"
                opacity: 0.6
            }
            ComboBox {
                id: modeBox
                model: controller.modeLabels
                currentIndex: controller.modes.indexOf(controller.mode)
                onActivated: controller.selectMode(currentIndex)
                enabled: !controller.running
                Layout.preferredWidth: 170
            }
        }
    }

    // One dialog pair reused by every input row: which one opens, and which
    // input receives the answer, is decided when Browse is clicked.
    property string pendingInput: ""

    FileDialog {
        id: fileDialog
        onAccepted: controller.setInput(root.pendingInput, selectedFile)
    }
    FolderDialog {
        id: dirDialog
        onAccepted: controller.setInput(root.pendingInput, selectedFolder)
    }
    FolderDialog {
        id: outputDialog
        title: "Select output directory"
        onAccepted: controller.setOutputDir(selectedFolder)
    }

    // One preview pane, used by both sides.  Nothing in it refers to
    // `controller` — the two values are bound from outside, where that
    // certainly resolves.
    component PreviewPane: RowLayout {
        property alias image: picture.source
        property alias info: details.text

        spacing: 10

        Rectangle {
            Layout.preferredWidth: 320
            Layout.preferredHeight: 320
            color: "transparent"
            border.color: "#9e9e9e"
            border.width: 1

            Image {
                id: picture
                anchors.fill: parent
                anchors.margins: 1
                fillMode: Image.PreserveAspectFit
                asynchronous: true
                cache: false
            }
            Label {
                anchors.centerIn: parent
                visible: picture.source == ""
                text: "Nothing to draw"
                opacity: 0.5
            }
        }

        ScrollView {
            Layout.fillWidth: true
            Layout.preferredHeight: 320
            TextArea {
                id: details
                readOnly: true
                wrapMode: TextArea.NoWrap
                font.family: "monospace"
                font.pixelSize: 11
            }
        }
    }

    function browseFor(spec) {
        root.pendingInput = spec.id
        if (spec.kind === "dir") {
            dirDialog.title = "Select " + spec.label.toLowerCase()
            dirDialog.open()
        } else {
            fileDialog.title = "Select " + spec.label.toLowerCase()
            fileDialog.nameFilters = spec.filter
                ? [spec.filter, "All files (*)"] : ["All files (*)"]
            fileDialog.open()
        }
    }

    ScrollView {
        anchors.fill: parent
        anchors.margins: 12
        contentWidth: availableWidth

        ColumnLayout {
            width: parent.width
            spacing: 14

            GroupBox {
                title: "Input"
                Layout.fillWidth: true
                ColumnLayout {
                    anchors.fill: parent
                    spacing: 6

                    // Only where the mode offers a choice: analysis reads a
                    // results directory and nothing else.
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        visible: controller.inputShapes.length > 1

                        Label { text: "Input"; Layout.preferredWidth: 90 }
                        ComboBox {
                            id: shapeBox
                            Layout.fillWidth: true
                            model: controller.inputShapes
                            currentIndex: controller.inputShape
                            onActivated: controller.selectInputShape(currentIndex)
                            enabled: !controller.running
                        }
                    }

                    Repeater {
                        model: controller.inputs

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8

                            Label {
                                text: modelData.label
                                Layout.preferredWidth: 90
                            }
                            TextField {
                                Layout.fillWidth: true
                                text: modelData.value
                                placeholderText: modelData.placeholder
                                onEditingFinished: controller.setInput(modelData.id, text)
                            }
                            Button {
                                text: "Browse…"
                                onClicked: root.browseFor(modelData)
                            }
                            // What this input's format has to contain.  Each
                            // shape reads a different one, and the wrong file
                            // fails deep inside a run rather than here.
                            Button {
                                id: contract
                                text: "!"
                                checkable: true
                                implicitWidth: 34
                                visible: modelData.help !== ""
                                ToolTip.text: modelData.help
                                ToolTip.visible: contract.checked
                                    || contractHover.hovered
                                HoverHandler { id: contractHover }
                            }
                        }
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8

                        Label { text: "Output dir"; Layout.preferredWidth: 90 }
                        TextField {
                            Layout.fillWidth: true
                            text: controller.outputDir
                            onEditingFinished: controller.setOutputDir(text)
                        }
                        Button { text: "Browse…"; onClicked: outputDialog.open() }
                    }
                }
            }

            GroupBox {
                title: "Parameters"
                Layout.fillWidth: true

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 6

                    Repeater {
                        model: controller.fields

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8

                            Label {
                                text: modelData.label
                                Layout.preferredWidth: 210
                                ToolTip.text: modelData.help
                                ToolTip.visible: modelData.help ? hover.hovered : false
                                HoverHandler { id: hover }
                            }

                            // Editors as siblings gated on `visible`, not a
                            // Loader with inline Components: those get a context
                            // where neither `controller` nor `modelData` resolves.
                            // QtQuick.Layouts skips invisible items, so only the
                            // relevant one takes space.
                            CheckBox {
                                visible: modelData.kind === "bool"
                                checked: controller.valueOf(modelData.id)
                                onToggled: controller.setValue(modelData.id, checked)
                            }

                            TextField {
                                visible: modelData.kind === "text"
                                Layout.fillWidth: true
                                text: controller.valueOf(modelData.id)
                                onEditingFinished: controller.setValue(modelData.id, text)
                            }

                            TextField {
                                visible: modelData.kind === "number"
                                    || modelData.kind === "integer"
                                Layout.fillWidth: true
                                text: controller.valueOf(modelData.id)
                                validator: DoubleValidator {}
                                onEditingFinished: controller.setValue(modelData.id, text)
                            }

                            RowLayout {
                                visible: modelData.kind === "size"
                                    || modelData.kind === "range"
                                Layout.fillWidth: true
                                TextField {
                                    Layout.preferredWidth: 90
                                    // Bindings of invisible siblings still
                                    // evaluate, so guard the indexing: on a
                                    // scalar field this would be undefined.
                                    text: modelData.kind === "size"
                                        || modelData.kind === "range"
                                        ? controller.valueOf(modelData.id)[0] : ""
                                    validator: DoubleValidator { bottom: 0 }
                                    onEditingFinished: controller.setSize(modelData.id, 0, text)
                                }
                                Label {
                                    text: modelData.kind === "range" ? "→" : "×"
                                }
                                TextField {
                                    Layout.preferredWidth: 90
                                    // Bindings of invisible siblings still
                                    // evaluate, so guard the indexing: on a
                                    // scalar field this would be undefined.
                                    text: modelData.kind === "size"
                                        || modelData.kind === "range"
                                        ? controller.valueOf(modelData.id)[1] : ""
                                    validator: DoubleValidator { bottom: 0 }
                                    onEditingFinished: controller.setSize(modelData.id, 1, text)
                                }
                                Item { Layout.fillWidth: true }
                            }
                        }
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 8

                Button {
                    text: controller.running ? "Cancel" : "Run"
                    highlighted: !controller.running
                    onClicked: controller.running ? controller.cancel() : controller.run()
                }
                ProgressBar {
                    Layout.fillWidth: true
                    from: 0
                    to: 100
                    value: controller.percent
                    indeterminate: controller.running && controller.percent <= 0
                }
                Label {
                    text: controller.running ? controller.percent.toFixed(0) + "%" : ""
                    Layout.preferredWidth: 44
                }
            }

            Label {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                text: controller.status
                color: controller.failed ? "#c0392b" : palette.text
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 8
                visible: controller.canPreview

                Label { text: "Preview"; opacity: 0.6 }
                Button {
                    text: "Original"
                    checkable: true
                    checked: controller.previewOriginal
                    onToggled: controller.setPreviewOriginal(checked)
                }
                Button {
                    text: "Synthetic"
                    checkable: true
                    checked: controller.previewSynthetic
                    // Only generate mode leaves the batch of synthetic
                    // networks a preview reads.
                    enabled: controller.canPreviewSynthetic
                    onToggled: controller.setPreviewSynthetic(checked)
                }
                Label {
                    Layout.fillWidth: true
                    text: controller.previewStatus
                    color: controller.previewFailed ? "#c0392b" : palette.text
                    elide: Text.ElideRight
                }
            }

            GroupBox {
                title: "Original network"
                Layout.fillWidth: true
                visible: controller.previewOriginal

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 8

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8

                        // Absent when the input came without an image: with
                        // nothing behind the network there is nothing to switch.
                        Button {
                            text: "Background"
                            checkable: true
                            checked: controller.showBackground
                            visible: controller.hasBackground
                            onToggled: controller.setShowBackground(checked)
                        }
                        Button {
                            text: "Network"
                            checkable: true
                            checked: controller.showNetwork
                            onToggled: controller.setShowNetwork(checked)
                        }
                        Label {
                            Layout.fillWidth: true
                            text: controller.originalNote
                            elide: Text.ElideRight
                            opacity: 0.7
                        }
                    }

                    PreviewPane {
                        Layout.fillWidth: true
                        image: controller.originalImage
                        info: controller.originalInfo
                    }
                }
            }

            GroupBox {
                title: "Synthetic network"
                Layout.fillWidth: true
                visible: controller.previewSynthetic

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 8

                    Label {
                        Layout.fillWidth: true
                        wrapMode: Text.WordWrap
                        text: controller.syntheticNote
                        opacity: 0.7
                    }

                    PreviewPane {
                        Layout.fillWidth: true
                        image: controller.syntheticImage
                        info: controller.syntheticInfo
                    }
                }
            }

            GroupBox {
                title: "Log"
                Layout.fillWidth: true
                Layout.preferredHeight: 220

                ScrollView {
                    anchors.fill: parent
                    TextArea {
                        id: logArea
                        readOnly: true
                        wrapMode: TextArea.NoWrap
                        font.family: "monospace"
                        font.pixelSize: 11
                        text: controller.logText
                        onTextChanged: cursorPosition = length
                    }
                }
            }
        }
    }
}
