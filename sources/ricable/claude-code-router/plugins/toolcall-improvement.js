module.exports = async function handle(req, res) {
  if (req?.body?.tools?.length) {
    req.body.system.push({
        type: "text",
        text: `## **Important Instruction:**  \nYou must use tools as frequently and accurately as possible to help the user solve their problem.\nPrioritize tool usage whenever it can enhance accuracy, efficiency, or the quality of the response.`
    })
  }
};
