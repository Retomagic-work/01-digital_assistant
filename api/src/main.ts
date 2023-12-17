import {NestFactory} from "@nestjs/core";
import {AppModule} from "./app.module";
import {ValidationPipe, VersioningType} from "@nestjs/common";
import {json, urlencoded} from "express";
import {DocumentBuilder, SwaggerModule} from "@nestjs/swagger";

async function bootstrap() {
    const PORT = Number(process.env.PORT) || 3000;
    const app: any = await NestFactory.create(AppModule, {cors: true});

    const config = new DocumentBuilder()
      .setTitle("LLM API")
      .setDescription("LLM API")
      .setVersion("1.0.0")
      .setContact("Alexey Snegirev", "https://github.com/viridius-hub", "smiledie-hub@yandex.ru")
      .build();

    const document = SwaggerModule.createDocument(app, config);

    SwaggerModule.setup("/docs", app, document, {
        explorer: true,
        customSiteTitle: "AiBooks API",
        customCss: ".swagger-ui .topbar { display: none }"
    });

    app.enableVersioning({
        type: VersioningType.URI
    });

    app.use(json({limit: "50mb"}));
    app.use(urlencoded({extended: true, limit: "50mb"}));
    app.set("trust proxy", true);

    app.useGlobalPipes(new ValidationPipe({transform: true}));

    await app.listen(PORT, () => console.log("Server started on port =", PORT));
}

bootstrap().catch(console.error);
